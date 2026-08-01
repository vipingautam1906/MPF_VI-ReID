import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math
from torchvision.models import resnet50, ResNet50_Weights

from utils.calc_acc import calc_acc
from models.modules.prototypegenerator import DynamicPrototypeGenerator
from models.modules.part_features import Part_Features

from losses.modality_aware_ranking_loss import MARLoss
from losses.cm_retrieval_loss import CMRetrievalLoss
from losses.cm_triplet import CrossModalityTripletLoss
from losses.triplet import TripletLoss

from models.modules.part_modality_remover import PartModalityRemover
from models.modules.modality_alignment import IntermediateModalityAlignment


class PMRNet(nn.Module):
    def __init__(self, num_classes, dataset='sysu', num_parts=6, parts = True, PMR=True, prototypes=True, MAR=True, alignment=True, p = 3.0, alpha=0.3):
        super(PMRNet, self).__init__()
        self.base_dim = 2048
        self.dataset = dataset
        self.num_parts = num_parts
        self.parts = parts
        self.PMR = PMR
        self.prototypes = prototypes
        self.MAR = MAR
        self.alignment = alignment
        self.p = p    
        self.alpha = alpha
       
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        backbone.layer4[0].conv2.stride = (1, 1)
        backbone.layer4[0].downsample[0].stride = (1, 1)
 
        self.current_epoch = 0        
        if self.PMR:
            self.part_modality_purifier = PartModalityRemover(self.base_dim, self.num_parts)
            
        # ---------------- Backbone ----------------
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # ---------------- Global BN neck ----------------
        self.bn_neck = nn.BatchNorm1d(self.base_dim)
        nn.init.constant_(self.bn_neck.bias, 0)
        self.bn_neck.bias.requires_grad_(False)

        # ---------------- Part Branch ----------------
        if self.parts:
            self.part_bns = nn.ModuleList([
                nn.BatchNorm1d(self.base_dim) for _ in range(self.num_parts)
            ])
            for bn in self.part_bns:
                nn.init.constant_(bn.bias, 0)
                bn.bias.requires_grad_(False)

            self.part_classifiers = nn.ModuleList([
                nn.Linear(self.base_dim, num_classes, bias=False)
                for _ in range(self.num_parts)
            ])
            self.extract_part_features = Part_Features(self.base_dim, self.num_parts, p=self.p)
        
        if self.prototypes:
            self.prototype_gen = DynamicPrototypeGenerator(self.base_dim, num_classes)
        
        self.classifier = nn.Linear(self.base_dim, num_classes, bias=False)

        # ---------------- Loss ----------------
        self.id_loss = nn.CrossEntropyLoss()
        self.mar_loss = MARLoss()
         
        if self.alignment:
            self.inter_align = IntermediateModalityAlignment(
                feat_dim=1024,   
                proj_dim=256
            )

        self.PMR_weight = 0.3 
        self.MAR_weight = 1.5  
        self.prototype_weight = 0.5 
        self.inter_align_weight = 0.1 
        
        print("=" * 100)
        print("-Modality Purification Framework-".center(100))
        print("=" * 100)

            
    def get_grl_alpha(self, epoch, total_epochs=300, max_alpha=1.0):
        p = epoch / total_epochs
        alpha = max_alpha * (2.0 / (1.0 + math.exp(-10 * p)) - 1.0) 
        return alpha

    def fuse_features(self, x, global_bn, fusion_mode='regdb'):
        global_bn = F.normalize(global_bn, dim=1)
        if not self.parts:
            return global_bn
        part_feats, _ = self.extract_part_features(x)
        
        reduced_parts = []
        for i in range(self.num_parts):
            reduced_parts.append(self.part_bns[i](part_feats[:, i]))
        reduced_parts = torch.stack(reduced_parts, dim=1) # B × P × D

        if self.PMR:
            part_features = self.part_modality_purifier(reduced_parts)
        else:
            part_features = reduced_parts
        
        part_features = F.normalize(part_features, dim=2)
        
        if fusion_mode in ('regdb', 'llcm'):
            return torch.cat([global_bn] + [part_features[:, i] for i in range(self.num_parts)],dim=1,)

        elif fusion_mode == 'sysu':
            parts_mean = part_features.mean(dim=1)
            return F.normalize(self.alpha * global_bn + (1 - self.alpha) * parts_mean,dim=1,)
        else:
            raise ValueError(f"Unknown fusion_mode '{fusion_mode}'.")
          
    def forward(self, inputs, labels=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x3 = self.layer3(x)    
        x4 = self.layer4(x3)   

        global_embeddings = (x4.clamp(min=1e-12) ** self.p).mean(dim=(-1, -2)) ** (1 / self.p)
        if self.training:
            return self.train_forward(x4, x3, global_embeddings, labels, **kwargs)
        else:
            feat = self.bn_neck(global_embeddings)
            feat = self.fuse_features(x4, feat, fusion_mode=self.dataset)
            inputs_flip = torch.flip(inputs, dims=[3])
            flipped_feat = self._forward_features(inputs_flip)
            feat = F.normalize((feat + flipped_feat) / 2, dim=1)
            return feat

    def _forward_features(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x3)
        global_embeddings = (x4.clamp(min=1e-12) ** self.p).mean(dim=(-1,-2)) ** (1/self.p)
        return self.fuse_features(x4, self.bn_neck(global_embeddings), fusion_mode=self.dataset)

    def train_forward(self, x, x3, global_embeddings, labels, **kwargs):
        cam_ids = kwargs.get('cam_ids')
        rgb_mask = (cam_ids == 1) | (cam_ids == 2) | (cam_ids == 4) | (cam_ids == 5)
        ir_mask  = (cam_ids == 3) | (cam_ids == 6)
        modality_labels = torch.zeros(len(labels), dtype=torch.long, device=labels.device)
        modality_labels[ir_mask] = 1
        
        if self.alignment:
            loss_inter_align = self.inter_align(x3, labels, modality_labels)
        metric = {}
        part_acc = {}

        # ---------------- Global branch ----------------
        bn_feat = self.bn_neck(global_embeddings)
        logits_fc = self.classifier(bn_feat)
        loss_id_fc = self.id_loss(logits_fc, labels)
        if self.MAR:
            loss_mar = self.mar_loss(bn_feat, labels, modality_labels)                  

        # ---------------- Prototype branch ----------------
        if self.prototypes:
            bn_feat_norm = F.normalize(bn_feat, dim=1)
            logits_proto = self.prototype_gen(
                bn_feat_norm, labels, cam_ids
            )
            loss_id_proto = self.id_loss(logits_proto.float(), labels)
            
        # ---------------- Part PMR branch ----------------
        if self.parts:
            part_feats, _ = self.extract_part_features(x)
            reduced_parts = []
            for i in range(self.num_parts):
                pf = part_feats[:, i]
                pf = self.part_bns[i](pf)
                reduced_parts.append(pf)
            reduced_parts = torch.stack(reduced_parts, dim=1)  # B × P × D

            if self.PMR:
                grl_alpha = self.get_grl_alpha(epoch=self.current_epoch, total_epochs=300)
                purified_parts_features, loss_PMR, loss_PMR_breakdown = self.part_modality_purifier(reduced_parts, modality_labels, labels, grl_alpha=grl_alpha)
            
            loss_part_id = 0.0
            for i in range(self.num_parts):
                if self.PMR:
                    pf    = purified_parts_features[:, i]
                else:
                    pf = reduced_parts[:, i]

                logit = self.part_classifiers[i](pf)
                p_loss = self.id_loss(logit, labels)
                loss_part_id += p_loss
                part_acc[f"acc_part_{i}"] = calc_acc(logit.detach(), labels)

            loss_part_id /= self.num_parts

        loss = loss_id_fc
        metric.update({"loss_id_fc":   loss_id_fc.detach()})
        metric.update({"acc":          calc_acc(logits_fc.detach(), labels)})
        if self.parts:
            loss = loss + loss_part_id
            metric.update(part_acc)
            metric.update({"loss_part_id": loss_part_id.detach()})
        if self.PMR:
            loss = loss + self.PMR_weight  * loss_PMR
            metric.update({"loss_PMR":  loss_PMR.detach()})
            metric.update({"loss_cov":     loss_PMR_breakdown["loss_cov"]})
            metric.update({"loss_mod":     loss_PMR_breakdown["loss_mod"]})
            metric.update({"loss_recon":   loss_PMR_breakdown["loss_recon"]})
        if self.prototypes:
            loss = loss + self.prototype_weight * loss_id_proto
            metric.update({"loss_id_proto": loss_id_proto.detach()})
            metric.update({"acc_proto":    calc_acc(logits_proto.detach(), labels)})
        if self.MAR:
            loss = loss + self.MAR_weight * loss_mar
            metric.update({"loss_mar": loss_mar.detach()})
        if self.alignment:
            loss = loss + self.inter_align_weight * loss_inter_align
            metric.update({"loss_inter_align": loss_inter_align.detach()})                
        return loss, metric