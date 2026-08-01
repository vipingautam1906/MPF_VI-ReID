import torch
import torch.nn as nn
import torchsort
from torch.nn import functional as F


class CMRetrievalLoss(nn.Module):

    def __init__(self):
        super(CMRetrievalLoss, self).__init__()
        print('CM Retrieval Loss')

    def forward(self, embeddings, id_labels, m_labels):
        embeddings = F.normalize(embeddings, dim=1)
        m_labels_unique = torch.unique(m_labels)
        m_num = len(m_labels_unique)

        embeddings_list = [embeddings[m_labels == m_label] for m_label in m_labels_unique]
        id_labels_list = [id_labels[m_labels == m_label] for m_label in m_labels_unique]

        cmr_loss = 0
        valid_m_count = 0
        for i in range(m_num):
            cur_m_embeddings = embeddings_list[i]
            cur_m_id_labels = id_labels_list[i]
            other_m_embeddings = torch.cat([embeddings_list[j] for j in range(len(m_labels_unique)) if j != i], dim=0)
            other_m_id_labels = torch.cat([id_labels_list[j] for j in range(len(m_labels_unique)) if j != i], dim=0)

            match_mask = cur_m_id_labels.unsqueeze(dim=1) == other_m_id_labels.unsqueeze(dim=0)
            mismatch_mask = ~match_mask
            match_num = match_mask.sum(dim=-1)
            mismatch_num = mismatch_mask.sum(dim=-1)

            remove_mask = (match_num == 0) | (match_num == len(other_m_id_labels))
            if remove_mask.all():
                continue
            cur_m_embeddings = cur_m_embeddings[~remove_mask]
            cur_m_id_labels = cur_m_id_labels[~remove_mask]
            match_mask = match_mask[~remove_mask]
            mismatch_mask = mismatch_mask[~remove_mask]
            match_num = match_num[~remove_mask]
            mismatch_num = mismatch_num[~remove_mask]

            dist_mat = F.cosine_similarity(cur_m_embeddings[:, None, :], other_m_embeddings[None, :, :], dim=-1)
            dist_mat = (1 - dist_mat) / 2

            predict_rank = torchsort.soft_rank(dist_mat, regularization="l2", regularization_strength=0.5)

            target_rank = torch.zeros_like(predict_rank)
            q_num, g_num = match_mask.shape

            target_rank[match_mask] = 1
            target_rank[mismatch_mask] = g_num

            cmr_loss += F.l1_loss(predict_rank, target_rank)

            valid_m_count += 1

        cmr_loss = cmr_loss / valid_m_count if valid_m_count > 0 else 0

        return cmr_loss



