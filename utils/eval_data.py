import os
import logging
import numpy as np
import torch
from torch.nn import functional as F
from .rerank import re_ranking, pairwise_distance
from collections import defaultdict

def get_gallery_names_sysu(perm, cams, ids, trial_id, num_shots=1):
    """
    A utility function designed specifically for constructing a list of gallery image file paths for the SYSU-MM01
    dataset in a cross-modality person re-identification task. This function takes a permutation array that organizes
    the data into camera views and identities, and then selects a specified number of shots (instances) for each
    identity from the desired cameras to compile the gallery set for a given trial. The output is a list of formatted
    strings that represent the file paths to the selected gallery images, adhering to the dataset's directory structure.

    Args:
    - perm (list): A nested list where each element corresponds to a camera view in the dataset.
      Each camera's list contains arrays of instance indices for each identity, organized by trials.
    - cams (list): A list of integers indicating which camera views to include in the gallery set.
      Camera numbers should match those used in the dataset.
    - ids (list): A list of integers specifying the identities to be included in the gallery set.
      Identity numbers should correspond to those in the dataset.
    - trial_id (int): The index of the trial for which to construct the gallery.
      This index is used to select specific instances from the permutation arrays,
      allowing for variability across different evaluation runs.
    - num_shots (int, optional): The number of shots to select for each identity from each camera view.

    Returns:
    - list: A list of strings, each representing the file path to an image selected for the gallery.
      Paths are formatted to match the directory structure of the SYSU-MM01 dataset.
    """

    names = []
    for cam in cams:
        cam_perm = perm[cam - 1][0].squeeze()
        for i in ids:
            if (i - 1) < len(cam_perm) and len(cam_perm[i - 1]) > 0:
                instance_id = cam_perm[i - 1][trial_id][:num_shots]
                names.extend(['cam{}/{:0>4d}/{:0>4d}'.format(cam, i, ins) for ins in instance_id.tolist()])
    return names


def get_unique(array):
    """
    A utility function that returns a sorted unique array of elements from the input array. It identifies all unique
    elements within the input array and selects their first occurrence, preserving the order of these unique elements
    based on their initial appearance in the input array. This function is particularly useful for processing arrays
    where the uniqueness and order of elements are essential, such as when filtering duplicate entries from lists of
    identifiers or categories without disrupting their original sequence.

    Args:
    - array (ndarray): An input array from which unique elements are to be extracted.

    Returns:
    - ndarray: A new array containing only the unique elements of the input array,
      sorted according to their first occurrence in the original array.
    """

    _, idx = np.unique(array, return_index=True)
    array_new = array[np.sort(idx)]
    return array_new


def get_cmc(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids):
    """
    A utility function for calculating the Cumulative Matching Characteristics (CMC) curve in person re-identification
    tasks. The CMC curve is a standard evaluation metric used to measure the performance of a re-identification model
    by determining the probability that a query identity appears in different sized candidate lists. This function
    processes the sorted indices of gallery samples for each query, excludes gallery samples captured by the same
    camera as the query to avoid camera bias, and computes the CMC curve based on the first correct match's position.

    Args:
    - sorted_indices (ndarray): An array of indices that sorts the gallery samples
      in ascending order of their distance to each query sample.
    - query_ids (ndarray): An array containing the identity labels of the query samples.
    - query_cam_ids (ndarray): An array containing the camera IDs associated with each query sample.
    - gallery_ids (ndarray): An array containing the identity labels of the gallery samples.
    - gallery_cam_ids (ndarray): An array containing the camera IDs associated with each gallery sample.

    Returns:
    - ndarray: The CMC curve represented as a 1D array where each element at index i indicates the probability
      that a query identity is correctly matched within the top-(i+1) ranks of the sorted gallery list.
    """

    gallery_unique_count = get_unique(gallery_ids).shape[0]
    match_counter = np.zeros((gallery_unique_count,))

    result = gallery_ids[sorted_indices]
    cam_locations_result = gallery_cam_ids[sorted_indices]

    valid_probe_sample_count = 0

    for probe_index in range(sorted_indices.shape[0]):
        # remove gallery samples from the same camera of the probe
        result_i = result[probe_index, :]
        result_i[np.equal(cam_locations_result[probe_index], query_cam_ids[probe_index])] = -1

        # remove the -1 entries from the label result
        result_i = np.array([i for i in result_i if i != -1])

        # remove duplicated id in "stable" manner - following the official test protocol in VI-ReID
        result_i_unique = get_unique(result_i)

        # match for probe i
        match_i = np.equal(result_i_unique, query_ids[probe_index])

        if np.sum(match_i) != 0:  # if there is true matching in gallery
            valid_probe_sample_count += 1
            match_counter += match_i

    rank = match_counter / valid_probe_sample_count
    cmc = np.cumsum(rank)
    return cmc


def get_mAP(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids):
    """
    A utility function for computing the mean Average Precision (mAP) for evaluating person re-identification methods.
    The mAP metric provides a single-figure measure of quality across recall levels, particularly useful in scenarios
    where the query identity appears multiple times in the gallery. This function iterates over each query, excludes
    gallery images captured by the same camera to prevent bias, and calculates the Average Precision (AP) for each
    query based on its matches in the gallery. The mAP is then obtained by averaging the APs across all queries.

    Args:
    - sorted_indices (ndarray): An array of indices that sorts the gallery samples
      in ascending order of their distance to each query sample.
    - query_ids (ndarray): An array containing the identity labels of the query samples.
    - query_cam_ids (ndarray): An array containing the camera IDs associated with each query sample.
    - gallery_ids (ndarray): An array containing the identity labels of the gallery samples.
    - gallery_cam_ids (ndarray): An array containing the camera IDs associated with each gallery sample.

    Returns:
    - float: The mean Average Precision (mAP) calculated across all query samples,
      representing the overall precision of the re-identification method at varying levels of recall.
    """

    result = gallery_ids[sorted_indices]
    cam_locations_result = gallery_cam_ids[sorted_indices]

    valid_probe_sample_count = 0
    avg_precision_sum = 0

    for probe_index in range(sorted_indices.shape[0]):
        # remove gallery samples from the same camera of the probe
        result_i = result[probe_index, :]
        result_i[cam_locations_result[probe_index, :] == query_cam_ids[probe_index]] = -1

        # remove the -1 entries from the label result
        result_i = np.array([i for i in result_i if i != -1])

        # match for probe i
        match_i = result_i == query_ids[probe_index]
        true_match_count = np.sum(match_i)

        if true_match_count != 0:  # if there is true matching in gallery
            valid_probe_sample_count += 1
            true_match_rank = np.where(match_i)[0]

            ap = np.mean(np.arange(1, true_match_count + 1) / (true_match_rank + 1))
            avg_precision_sum += ap

    mAP = avg_precision_sum / valid_probe_sample_count
    return mAP


def get_mINP(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids):
    """
    A utility function designed for evaluating the mean Inverse Negative Penalty (mINP) across all query samples in
    a person re-identification method. The mINP metric focuses on the hardest positive sample's position in the ranked
    list of gallery samples for each query, providing insight into the system's ability to recall all relevant instances
    of an identity. This function computes the INP for each query by excluding gallery samples captured by the same
    camera as the query, identifying the rank position of the farthest correct match, and calculating the INP based on
    this position. The mean INP is then derived by averaging the INP scores across all valid queries.

    Args:
    - sorted_indices (ndarray): An array of indices that sorts the gallery samples
      in ascending order of their distance to each query sample.
    - query_ids (ndarray): An array containing the identity labels of the query samples.
    - query_cam_ids (ndarray): An array containing the camera IDs associated with each query sample.
    - gallery_ids (ndarray): An array containing the identity labels of the gallery samples.
    - gallery_cam_ids (ndarray): An array containing the camera IDs associated with each gallery sample.

    Returns:
    - float: The mean Inverse Negative Penalty (mINP) calculated across all queries, reflecting the method's
      effectiveness in retrieving relevant matches from gallery, particularly the most challenging matches to identify.
    """

    result = gallery_ids[sorted_indices]
    cam_locations_result = gallery_cam_ids[sorted_indices]

    valid_probe_sample_count = 0
    INP_sum = 0

    for probe_index in range(sorted_indices.shape[0]):
        # remove gallery samples from the same camera of the probe
        result_i = result[probe_index, :]
        result_i[np.equal(cam_locations_result[probe_index], query_cam_ids[probe_index])] = -1

        # remove the -1 entries from the label result
        result_i = np.array([i for i in result_i if i != -1])

        # match for probe i
        match_i = result_i == query_ids[probe_index]
        true_match_count = np.sum(match_i)

        if true_match_count != 0:  # if there is true matching in gallery
            valid_probe_sample_count += 1
            true_match_rank = np.where(match_i)[0]
            hardest_match_pos = true_match_rank[-1] + 1

            NP = (hardest_match_pos - true_match_count) / hardest_match_pos
            INP = 1 - NP

            INP_sum += INP

    mINP = INP_sum / valid_probe_sample_count

    return mINP


def get_cmc_mAP_mINP(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids):
    """
    A comprehensive utility function designed to compute three key metrics simultaneously for evaluating the performance
    of person re-identification methods: Cumulative Matching Characteristics (CMC), mean Average Precision (mAP), and
    mean Inverse Negative Penalty (mINP). This function integrates the processes of calculating these metrics into a
    single operation, optimizing the evaluation workflow for the person re-identification task.

    Args:
    - sorted_indices (ndarray): Indices sorting the gallery samples by ascending similarity to each query sample.
    - query_ids (ndarray): Identity labels for the query samples.
    - query_cam_ids (ndarray): Camera IDs from which each query sample was captured.
    - gallery_ids (ndarray): Identity labels for the gallery samples.
    - gallery_cam_ids (ndarray): Camera IDs from which each gallery sample was captured.

    Returns:
    - tuple: A tuple containing three elements:
      - cmc (ndarray): The CMC curve as a 1D array.
      - mAP (float): The mean Average Precision score.
      - mINP (float): The mean Inverse Negative Penalty score.
    """

    gallery_unique_count = get_unique(gallery_ids).shape[0]
    match_counter = np.zeros((gallery_unique_count,))

    result = gallery_ids[sorted_indices]
    cam_locations_result = gallery_cam_ids[sorted_indices]

    valid_probe_sample_count = 0
    avg_precision_sum = 0
    INP_sum = 0

    for probe_index in range(sorted_indices.shape[0]):
        # remove gallery samples from the same camera of the probe
        result_i = result[probe_index, :]
        result_i[np.equal(cam_locations_result[probe_index], query_cam_ids[probe_index])] = -1

        # remove the -1 entries from the label result
        result_i = np.array([i for i in result_i if i != -1])

        # remove duplicated id in "stable" manner - following the official test protocol in VI-ReID
        result_i_unique = get_unique(result_i)

        # match for probe i
        match_i = result_i == query_ids[probe_index]
        true_match_count = np.sum(match_i)
        match_i_unique = np.equal(result_i_unique, query_ids[probe_index])

        if true_match_count != 0:  # if there is true matching in gallery
            valid_probe_sample_count += 1
            if match_counter.shape != match_i_unique.shape:
                sub_num = match_counter.shape[0] - match_i_unique.shape[0]
                match_i_unique = np.hstack([match_i_unique, [False] * sub_num])
            match_counter += match_i_unique
            true_match_rank = np.where(match_i)[0]
            ap = np.mean(np.arange(1, true_match_count + 1) / (true_match_rank + 1))
            avg_precision_sum += ap
            hardest_match_pos = true_match_rank[-1] + 1
            NP = (hardest_match_pos - true_match_count) / hardest_match_pos
            INP = 1 - NP
            INP_sum += INP

    rank = match_counter / valid_probe_sample_count
    cmc = np.cumsum(rank)
    mAP = avg_precision_sum / valid_probe_sample_count
    mINP = INP_sum / valid_probe_sample_count
    return cmc, mAP, mINP


def eval_sysu(query_feats, query_ids, query_cam_ids, query_img_paths,
              gallery_feats, gallery_ids, gallery_cam_ids, gallery_img_paths,
              perm, mode='all', num_shots=1, num_trials=10, rerank=False):
    return eval_sysu_base(query_feats, query_ids, query_cam_ids, query_img_paths,
                          gallery_feats, gallery_ids, gallery_cam_ids, gallery_img_paths,
                          perm, mode, num_shots, num_trials, rerank)


def eval_sysu_base(query_feats, query_ids, query_cam_ids, query_img_paths,
                   gallery_feats, gallery_ids, gallery_cam_ids, gallery_img_paths,
                   perm, mode='all', num_shots=1, num_trials=10, rerank=False):
    assert mode in ['indoor', 'all']

    gallery_cams = [1, 2] if mode == 'indoor' else [1, 2, 4, 5]

    # cam2 and cam3 are in the same location
    query_cam_ids[np.equal(query_cam_ids, 3)] = 2
    query_feats = F.normalize(query_feats, dim=1)

    gallery_indices = np.in1d(gallery_cam_ids, gallery_cams)

    gallery_feats = gallery_feats[gallery_indices]
    gallery_feats = F.normalize(gallery_feats, dim=1)
    gallery_ids = gallery_ids[gallery_indices]
    gallery_cam_ids = gallery_cam_ids[gallery_indices]
    gallery_img_paths = gallery_img_paths[gallery_indices]
    gallery_names = np.array(['/'.join(os.path.splitext(path)[0].split('/')[-3:]) for path in gallery_img_paths])

    gallery_id_set = np.unique(gallery_ids)

    r1, r5, r10, r20, mAP, mINP = 0, 0, 0, 0, 0, 0
    rank_results = []
    for t in range(num_trials):
        names = get_gallery_names_sysu(perm, gallery_cams, gallery_id_set, t, num_shots)
        flag = np.in1d(gallery_names, names)

        g_feats = gallery_feats[flag]
        g_ids = gallery_ids[flag]
        g_cam_ids = gallery_cam_ids[flag]
        g_img_paths = gallery_img_paths[flag]

        # dist_mat = pairwise_distance(query_feats, g_feats)  # A
        # dist_mat = -torch.mm(query_feats, g_feats.permute(1, 0))  # B
        # if rerank:
        #     dist_mat = re_ranking(query_feats, gallery_feats)
        # else:
        #     dist_mat = (-torch.mm(query_feats, gallery_feats.t())).cpu().numpy()

        if rerank:
            dist_mat = re_ranking(query_feats, g_feats)
        else:
            dist_mat = (-torch.mm(query_feats, g_feats.t())).cpu().numpy()

        # When using normalization on extracted features, these two distance measures are equivalent (A = 2 + 2 * B)
        # B is a little faster than A
        sorted_indices = np.argsort(dist_mat, axis=1)

        cur_rank_results = dict()
        cur_rank_results['query_ids'] = query_ids
        cur_rank_results['query_img_paths'] = query_img_paths
        cur_rank_results['gallery_ids'] = g_ids
        cur_rank_results['gallery_img_paths'] = g_img_paths
        cur_rank_results['dist_mat'] = dist_mat
        cur_rank_results['sorted_indices'] = sorted_indices
        rank_results.append(cur_rank_results)

        cur_cmc, cur_mAP, cur_mINP = get_cmc_mAP_mINP(sorted_indices, query_ids, query_cam_ids, g_ids, g_cam_ids)
        r1 += cur_cmc[0]
        r5 += cur_cmc[4]
        r10 += cur_cmc[9]
        r20 += cur_cmc[19]
        mAP += cur_mAP
        mINP += cur_mINP

    r1 = r1 / num_trials * 100
    r5 = r5 / num_trials * 100
    r10 = r10 / num_trials * 100
    r20 = r20 / num_trials * 100
    mAP = mAP / num_trials * 100
    mINP = mINP / num_trials * 100

    logger = logging.getLogger('MCJA')
    logger.info('-' * 150)
    perf = '{} num-shot:{} r1 precision = {:.2f} , r10 precision = {:.2f} , r20 precision = {:.2f} , ' \
           'mAP = {:.2f} , mINP = {:.2f}'
    logger.info(perf.format(mode, num_shots, r1, r10, r20, mAP, mINP))
    logger.info('-' * 150)

    return r1, r5, r10, r20, mAP, mINP, rank_results

def eval_regdb(query_feats, query_ids, query_cam_ids, query_img_paths,
               gallery_feats, gallery_ids, gallery_cam_ids, gallery_img_paths, mode='i2v', rerank=False):
    return eval_regdb_base(query_feats, query_ids, query_cam_ids, query_img_paths,
                           gallery_feats, gallery_ids, gallery_cam_ids, gallery_img_paths, mode, rerank)


def eval_regdb_base(query_feats, query_ids, query_cam_ids, query_img_paths,
                    gallery_feats, gallery_ids, gallery_cam_ids, gallery_img_paths, mode='i2v', rerank=False):
    assert mode in ['i2v', 'v2i']

    gallery_feats = F.normalize(gallery_feats, dim=1)
    query_feats = F.normalize(query_feats, dim=1)

    # dist_mat = pairwise_distance(query_feats, gallery_feats)
    #dist_mat = -torch.mm(query_feats, gallery_feats.t())
    if rerank:
        dist_mat = re_ranking(query_feats, gallery_feats, eval_type=False)
    else:
        dist_mat = (-torch.mm(query_feats, gallery_feats.t())).cpu().numpy()
    sorted_indices = np.argsort(dist_mat, axis=1)

    rank_results = []
    cur_rank_results = dict()
    cur_rank_results['query_ids'] = query_ids
    cur_rank_results['query_img_paths'] = query_img_paths
    cur_rank_results['gallery_ids'] = gallery_ids
    cur_rank_results['gallery_img_paths'] = gallery_img_paths
    cur_rank_results['dist_mat'] = dist_mat
    cur_rank_results['sorted_indices'] = sorted_indices
    rank_results.append(cur_rank_results)

    # mAP = get_mAP(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids)
    # cmc = get_cmc(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids)
    # mINP = get_mINP(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids)
    cmc, mAP, mINP = get_cmc_mAP_mINP(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids)

    r1 = cmc[0]
    r5 = cmc[4]
    r10 = cmc[9]
    r20 = cmc[19]

    r1 = r1 * 100
    r5 = r5 * 100
    r10 = r10 * 100
    r20 = r20 * 100
    mAP = mAP * 100
    mINP = mINP * 100

    logger = logging.getLogger('MCJA')
    logger.info('-' * 150)
    perf = 'r1 precision = {:.2f} , r10 precision = {:.2f} , r20 precision = {:.2f} , ' \
           'mAP = {:.2f} , mINP = {:.2f}'
    logger.info(perf.format(r1, r10, r20, mAP, mINP))
    logger.info('-' * 150)

    return r1, r5, r10, r20, mAP, mINP, rank_results


def eval_regdb_mser(query_feats_list, query_ids_list, query_cam_ids_list, query_img_paths_list,
                    gallery_feats_list, gallery_ids_list, gallery_cam_ids_list, gallery_img_paths_list, mode='i2v'):
    """
    A function crafted for evaluating VI-ReID models on the RegDB dataset using the Multi-Spectral Enhanced Ranking
    (MSER) strategy, tailored specifically for the thermal-to-visible (t2v or i2v) or visible-to-thermal (v2t or v2i)
    matching scenarios. The MSER strategy applies a re-ranking mechanism to enhance the initial distance matrix
    computation, utilizing multiple spectral representations. This function performs normalization on the feature
    vectors of both query and gallery sets, computes the distance matrix with MSER re-ranking, and evaluates the
    model based on re-identification metrics.

    Args:
    - query_feats_list (List[Tensor]): List of tensors representing features of query images.
    - query_ids_list (List[Tensor]): List of tensors containing the identity labels of the query images.
    - query_cam_ids_list (List[Tensor]): List of tensors with camera IDs from which each query image was captured.
    - query_img_paths_list (List[ndarray]): List of ndarrays holding the file paths for each query image.
    - gallery_feats_list (List[Tensor]): List of tensors representing the feature vectors of gallery images.
    - gallery_ids_list (List[Tensor]): List of tensors containing the identity labels of the gallery images.
    - gallery_cam_ids_list (List[Tensor]): List of tensors with camera IDs from which each gallery image was captured.
    - gallery_img_paths_list (List[ndarray]): List of ndarrays holding the file paths for each gallery image.
    - mode (str): The evaluation mode, either 'i2v' (infrared to visible) or 'v2i' (visible to infrared),
      dictating the direction of matching between query and gallery sets.

    Returns:
    - tuple: A tuple containing metrics of rank-1, rank-5, rank-10, rank-20 precision, mean Average Precision (mAP),
      and mean Inverse Negative Penalty (mINP), along with detailed rank results for further analysis.
    """

    list_num = len(query_feats_list)
    for c in range(list_num):
        gallery_feats_list[c] = F.normalize(gallery_feats_list[c], dim=1)
        query_feats_list[c] = F.normalize(query_feats_list[c], dim=1)

    dist_mat = mser_rerank(query_feats_list, gallery_feats_list,
                           k1=50, k2=10, lambda_value=0.2, eval_type=False, mode=mode)
    sorted_indices = np.argsort(dist_mat, axis=1)

    rank_results = []
    cur_rank_results = dict()
    cur_rank_results['query_ids'] = query_ids_list[0]
    cur_rank_results['query_img_paths'] = query_img_paths_list[0]
    cur_rank_results['gallery_ids'] = gallery_ids_list[0]
    cur_rank_results['gallery_img_paths'] = gallery_img_paths_list[0]
    cur_rank_results['dist_mat'] = dist_mat[0]
    cur_rank_results['sorted_indices'] = sorted_indices[0]
    rank_results.append(cur_rank_results)

    # mAP = get_mAP(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids)
    # cmc = get_cmc(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids)
    # mINP = get_mINP(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids)
    cmc, mAP, mINP = get_cmc_mAP_mINP(sorted_indices,
                                      query_ids_list[0], query_cam_ids_list[0],
                                      gallery_ids_list[0], gallery_cam_ids_list[0])

    r1 = cmc[0]
    r5 = cmc[4]
    r10 = cmc[9]
    r20 = cmc[19]

    r1 = r1 * 100
    r5 = r5 * 100
    r10 = r10 * 100
    r20 = r20 * 100
    mAP = mAP * 100
    mINP = mINP * 100

    logger = logging.getLogger('MCJA')
    logger.info('-' * 150)
    perf = '[MSER] r1 precision = {:.2f} , r10 precision = {:.2f} , r20 precision = {:.2f} , ' \
           'mAP = {:.2f} , mINP = {:.2f}'
    logger.info(perf.format(r1, r10, r20, mAP, mINP))
    logger.info('-' * 150)

    return r1, r5, r10, r20, mAP, mINP, rank_results


def eval_llcm(query_feats, query_ids, query_cam_ids, query_img_paths,
              gallery_feats, gallery_ids, gallery_cam_ids, gallery_img_paths,
              mode='i2v', rerank=False):

    assert mode in ['i2v', 'v2i']
    """
        Ref: https://github.com/ZYK100/LLCM/blob/main/DEEN/eval_metrics.py
    """
    query_feats = F.normalize(query_feats, dim=1)
    gallery_feats = F.normalize(gallery_feats, dim=1)

    query_ids = np.asarray(query_ids)
    query_cam_ids = np.asarray(query_cam_ids)
    gallery_ids = np.asarray(gallery_ids)
    gallery_cam_ids = np.asarray(gallery_cam_ids)
    gallery_img_paths = np.asarray(gallery_img_paths)

    if rerank:
        dist_mat = re_ranking(query_feats, gallery_feats)
    else:
        dist_mat = (-torch.mm(query_feats, gallery_feats.t())).cpu().numpy()

    num_q, num_g = dist_mat.shape
    max_rank = min(20, num_g)

    indices = np.argsort(dist_mat, axis=1)
    pred_label = gallery_ids[indices]
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis]).astype(np.int32)

    new_all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.

    for q_idx in range(num_q):
        q_pid = query_ids[q_idx]
        q_cam = query_cam_ids[q_idx]
        order = indices[q_idx]
        remove = (gallery_ids[order] == q_pid) & (gallery_cam_ids[order] == q_cam)
        keep = np.invert(remove)

        # ---------- CMC ----------
        new_cmc = pred_label[q_idx][keep]
        _, first_index = np.unique(new_cmc, return_index=True)
        new_cmc = new_cmc[np.sort(first_index)]

        new_match = (new_cmc == q_pid).astype(np.int32)
        new_cmc = new_match.cumsum()

        if len(new_cmc) < max_rank:
            new_cmc = np.pad(new_cmc,
                             (0, max_rank - len(new_cmc)),
                             mode='edge')

        new_all_cmc.append(new_cmc[:max_rank])

        # ---------- AP / INP ----------
        orig_cmc = matches[q_idx][keep]

        if not np.any(orig_cmc):
            continue

        cmc = orig_cmc.cumsum()

        pos_idx = np.where(orig_cmc == 1)[0]
        pos_max = pos_idx[-1]
        inp = cmc[pos_max] / (pos_max + 1.0)
        all_INP.append(inp)

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = np.asarray([x / (i + 1.) for i, x in enumerate(tmp_cmc)])
        AP = (tmp_cmc * orig_cmc).sum() / num_rel
        all_AP.append(AP)

        num_valid_q += 1.

    assert num_valid_q > 0

    new_all_cmc = np.asarray(new_all_cmc).astype(np.float32)
    cmc = new_all_cmc.sum(0) / num_valid_q

    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)

    r1 = cmc[0] * 100
    r5 = cmc[4] * 100
    r10 = cmc[9] * 100
    r20 = cmc[19] * 100
    mAP *= 100
    mINP *= 100

    rank_results = [{
        'query_ids': query_ids,
        'query_img_paths': query_img_paths,
        'gallery_ids': gallery_ids,
        'gallery_img_paths': gallery_img_paths,
        'dist_mat': dist_mat,
        'sorted_indices': indices
    }]

    logger = logging.getLogger('MCJA')
    logger.info('-' * 150)
    logger.info(
        '[LLCM-{}] r1 = {:.2f} , r10 = {:.2f} , r20 = {:.2f} , mAP = {:.2f} , mINP = {:.2f}'.format(
            mode, r1, r10, r20, mAP, mINP
        )
    )
    logger.info('-' * 150)

    return r1, r5, r10, r20, mAP, mINP, rank_results


