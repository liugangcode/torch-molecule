import torch
from torch_geometric.data import Data
from torch.distributions.beta import Beta
from torch_geometric.loader import DataLoader
from typing import List, Tuple, Dict, Optional

def get_sample_probs(counts: torch.Tensor) -> torch.Tensor:
    """Calculate sampling probabilities based on counts."""
    idx = torch.argsort(counts.view(-1), descending=False)
    sample_rate = counts[torch.flipud(idx)] / counts.max()
    probs = torch.zeros_like(sample_rate)
    probs[idx] = sample_rate
    return probs

def get_datapoint_list(subset_data: List[Data], new_label_all: Optional[torch.Tensor] = None) -> List[Data]:
    """Convert dataset into list of Data objects with optional new labels."""
    datapoint_list = []
    for idx, datapoint in enumerate(subset_data):
        g = Data(
            edge_index=datapoint.edge_index,
            edge_attr=datapoint.edge_attr,
            x=datapoint.x,
            y=torch.tensor([[new_label_all[idx]]]) if new_label_all is not None else datapoint.y
        )
        
        for attr in ['morgan', 'maccs']:
            if hasattr(datapoint, attr):
                setattr(g, attr, getattr(datapoint, attr))
                
        datapoint_list.append(g)
    return datapoint_list

def mean_by_groups(sample_rep: torch.Tensor, groups: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate mean representations and counts per group."""
    weight = torch.zeros(groups.max() + 1, sample_rep.shape[0], device=sample_rep.device)
    weight[groups, torch.arange(sample_rep.shape[0])] = 1
    group_count = weight.sum(dim=1)
    weight = torch.nn.functional.normalize(weight, p=1, dim=1)
    mean = torch.mm(weight, sample_rep)
    valid_indices = torch.nonzero(group_count > 0).squeeze()
    return mean[valid_indices], group_count[valid_indices]

def build_selection_dataset(
    model: torch.nn.Module,
    labeled_dataset: Data,
    unlbl_dataset: Data,
    batch_size: int,
    num_anchor: int,
    threshold: float,
    device: torch.device,
    label_logscale: bool = False
) -> DataLoader:
    """Build selection dataset using model predictions and uncertainty."""
    labeled_targets = torch.tensor([g.y for g in labeled_dataset])
    unlabel_idx = torch.arange(len(unlbl_dataset))

    # Create data loaders
    unlbl_loader = DataLoader(unlbl_dataset, batch_size=batch_size, shuffle=False)
    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=False)
    
    # Get model predictions and uncertainty for unlabeled data
    model.eval()
    unlbl_pred, unlbl_env_vars = [], []
    
    with torch.no_grad():
        for batch in unlbl_loader:
            if batch.x.shape[0] <= 1:
                continue
                
            batch = batch.to(device)
            output = model(batch)
            pred = output['prediction']
            var = output['variance']
            unlbl_env_vars.append(var.detach())
            unlbl_pred.append(pred.detach())

    unlbl_pred = torch.cat(unlbl_pred, dim=0).cpu().view(-1)
    unlbl_env_vars = torch.cat(unlbl_env_vars, dim=0).cpu().view(-1)

    # Sort by uncertainty
    var_asc_idx = torch.argsort(unlbl_env_vars)
    unlbl_env_vars = unlbl_env_vars[var_asc_idx]
    unlbl_pred = unlbl_pred[var_asc_idx]
    unlabel_idx = unlabel_idx[var_asc_idx]

    # Get uncertainty for labeled data
    labeled_env_vars = []
    with torch.no_grad():
        for batch in labeled_loader:
            if batch.x.shape[0] <= 1:
                continue
                
            batch = batch.to(device)
            output = model(batch)
            var = output['variance']
            labeled_env_vars.append(var.detach())
            
    labeled_env_vars = torch.cat(labeled_env_vars, dim=0).cpu().view(-1)

    # Filter by uncertainty threshold
    uncertainty_masks = unlbl_env_vars <= torch.quantile(labeled_env_vars, threshold)
    unlbl_pred = unlbl_pred[uncertainty_masks]
    unlabel_idx = unlabel_idx[uncertainty_masks]

    # Create label boundaries
    start, end = labeled_targets.min(), labeled_targets.max()
    if label_logscale:
        start, end = torch.log10(start), torch.log10(end)
    boundaries = torch.linspace(start, end, steps=num_anchor + 1)
    if label_logscale:
        boundaries = torch.pow(10, boundaries)

    # Assign samples to buckets
    bucket_ids = torch.bucketize(labeled_targets.view(-1), boundaries)
    bucket_ids = torch.clamp(bucket_ids, min=1, max=len(boundaries) - 1)
    unique_buckets, bucket_counts = torch.unique(bucket_ids, sorted=True, return_counts=True)
    bucket_centers = (boundaries[unique_buckets - 1] + boundaries[unique_buckets]) / 2
    
    # Sample from each bucket
    sampling_probs = get_sample_probs(bucket_counts)
    new_idx_all, new_label_all = [], []
    
    for idx, anchor in enumerate(bucket_centers):
        upper_idx = torch.nonzero(boundaries > anchor)[0]
        lower_idx = torch.nonzero(boundaries < anchor)[-1]
        width = boundaries[upper_idx] - boundaries[lower_idx]
        
        valid_mask = torch.logical_and(
            unlbl_pred >= anchor - width/2,
            unlbl_pred < anchor + width/2
        )
        
        num_picked = min(
            int(valid_mask.sum() * sampling_probs[idx]),
            bucket_counts.max()
        )
        
        if num_picked > 0:
            label_dist = torch.abs(anchor - unlbl_pred[valid_mask])
            idx_sorted = torch.argsort(label_dist)[:num_picked]
            new_idx = unlabel_idx[valid_mask][idx_sorted]
            new_idx_all.append(new_idx)
            new_label_all.append(torch.full((num_picked,), anchor))

    if new_idx_all:
        new_idx_all = torch.cat(new_idx_all)
        new_label_all = torch.cat(new_label_all)
        
        # Create final dataset
        new_trainset = get_datapoint_list(labeled_dataset)
        new_unlbl_data = [unlbl_dataset[idx.item()] for idx in new_idx_all]
        
        new_trainset.extend(get_datapoint_list(new_unlbl_data, new_label_all))
        
        return DataLoader(new_trainset, batch_size=batch_size, shuffle=True)
    
    return DataLoader(get_datapoint_list(labeled_dataset), batch_size=batch_size, shuffle=True)

def build_augmentation_dataset(
    model: torch.nn.Module,
    labeled_dataset: Data,
    unlbl_dataset: Data,
    batch_size: int,
    num_anchor: int,
    device: torch.device,
    label_logscale: bool = False,
) -> Dict[str, torch.Tensor]:
    """Build augmentation dataset using mixup strategy."""
    # Create data loaders
    unlbl_loader = DataLoader(unlbl_dataset, batch_size=batch_size, shuffle=False)
    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=False)
    
    # Get labeled data representations
    model.eval()
    labeled_targets, labeled_reps = [], []
    
    with torch.no_grad():
        for batch in labeled_loader:
            if batch.x.shape[0] <= 1:
                continue
                
            batch = batch.to(device)
            output = model(batch)
            labeled_targets.append(batch.y.view(output['prediction'].shape))
            labeled_reps.append(output['representation'])

    labeled_targets = torch.cat(labeled_targets, dim=0)
    labeled_reps = torch.cat(labeled_reps, dim=0)
    
    # Get unlabeled data representations
    unlbl_pred, unlabeled_reps = [], []
    
    with torch.no_grad():
        for batch in unlbl_loader:
            if batch.x.shape[0] <= 1:
                continue
                
            batch = batch.to(device)
            output = model(batch)
            unlbl_pred.append(output['prediction'])
            unlabeled_reps.append(output['representation'])

    unlbl_pred = torch.cat(unlbl_pred, dim=0)
    unlabeled_reps = torch.cat(unlabeled_reps, dim=0)

    # Combine all representations
    all_reps = torch.cat([labeled_reps, unlabeled_reps])
    all_labels = torch.cat([labeled_targets, unlbl_pred])

    # Create label boundaries
    start, end = labeled_targets.min(), labeled_targets.max()
    if label_logscale:
        start, end = torch.log10(start), torch.log10(end)
    boundaries = torch.linspace(start, end, steps=num_anchor + 1)
    if label_logscale:
        boundaries = torch.pow(10, boundaries)
    boundaries = boundaries.to(device)

    # Assign samples to buckets
    bucket_ids = torch.bucketize(labeled_targets.view(-1), boundaries)
    bucket_ids = torch.clamp(bucket_ids, min=1, max=len(boundaries) - 1)
    unique_buckets = torch.unique(bucket_ids, sorted=True)
    bucket_centers = (boundaries[unique_buckets - 1] + boundaries[unique_buckets]) / 2
    
    # Calculate bucket statistics
    bucket_rep, bucket_count = mean_by_groups(labeled_reps, bucket_ids - 1)
    buckets_preds_dist = torch.cdist(bucket_centers.view(-1, 1), all_labels.view(-1, 1), p=2)
    rank_per_buckets = torch.argsort(buckets_preds_dist, dim=1)

    # Calculate sampling probabilities
    sampling_probs = get_sample_probs(bucket_count)
    samples_per_bucket = torch.clamp(
        torch.ceil(sampling_probs * bucket_count.max()).to(torch.int),
        max=min(rank_per_buckets.size(1), 100)
    )
    
    max_samples = samples_per_bucket.max()
    sample_indices = rank_per_buckets[:, :max_samples].contiguous().view(-1)

    # Perform mixup augmentation
    beta_dist = Beta(torch.tensor([5.]).to(device), torch.tensor([1.]).to(device))
    lambdas = beta_dist.sample((bucket_rep.size(0),)).view(-1, 1)
    lambdas = torch.max(torch.cat([lambdas, 1 - lambdas], dim=1), dim=1).values

    # Create mixed representations and labels
    mixed_reps = []
    mixed_labels = []
    
    for idx, (bucket_r, center, num_samples) in enumerate(zip(bucket_rep, bucket_centers, samples_per_bucket)):
        if num_samples == 0:
            continue
            
        lambda_idx = lambdas[idx]
        samples_r = all_reps[sample_indices[idx * max_samples:idx * max_samples + num_samples]]
        samples_l = all_labels[sample_indices[idx * max_samples:idx * max_samples + num_samples]]
        
        mixed_r = lambda_idx * bucket_r + (1 - lambda_idx) * samples_r
        mixed_l = lambda_idx * center + (1 - lambda_idx) * samples_l
        
        mixed_reps.append(mixed_r)
        mixed_labels.append(mixed_l)

    return {
        'representations': torch.cat(mixed_reps),
        'labels': torch.cat(mixed_labels)
    }