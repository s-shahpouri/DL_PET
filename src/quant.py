import numpy as np
import nibabel as nib
from math import sqrt, log10
from skimage.metrics import structural_similarity as ssim

def mean_error(predicted, reference):
    return np.mean(predicted - reference)

def mean_absolute_error(predicted, reference):
    return np.mean(np.abs(predicted - reference))

def relative_error(predicted, reference):

    re = np.mean((predicted - reference) / (reference )) * 100
    return re

def absolute_relative_error(predicted, reference):
    # Create a mask for pixels in the reference image above the threshold

    # Calculate the absolute relative error using the masked pixels
    are = np.mean(np.abs(predicted - reference) / reference) * 100
    
    return are

def rmse(predicted, reference):
    return sqrt(np.mean((predicted - reference) ** 2))

def psnr(predicted, reference, peak):
    mse = np.mean((predicted - reference) ** 2)
    return 20 * log10(peak / sqrt(mse))

def calculate_ssim(predicted, reference):
    return ssim(predicted, reference, data_range=reference.max() - reference.min())

def load_nifti_image(path):
    """Load a NIfTI image and return its data as a NumPy array."""
    return nib.load(path).get_fdata()

import numpy as np

def masked_SUV_img(nac_path, predicted_path, reference_path, nac_factor, mac_factor, mask_val):
    predicted_img = load_nifti_image(predicted_path) * mac_factor
    reference_img = load_nifti_image(reference_path) * mac_factor
    nac_img = load_nifti_image(nac_path) * nac_factor

    if predicted_img.size == 0 or reference_img.size == 0 or nac_img.size == 0:
        raise ValueError("One or more images did not load correctly or are empty.")

    mask = reference_img > mask_val
    
    # Apply the mask and replace unmasked values with NaN
    masked_predicted_img = np.where(mask, predicted_img, np.nan)
    masked_reference_img = np.where(mask, reference_img, np.nan)
    masked_nac_img = np.where(mask, nac_img, np.nan)

    # Debug: Print how many values are being masked
    # print(f"Valid data points after masking: {np.count_nonzero(~np.isnan(masked_predicted_img))}")

    return masked_nac_img, masked_predicted_img, masked_reference_img

def calculate_metrics_for_pair(nac_path, predicted_path, reference_path, nac_factor, mac_factor, mask_val):
    """
    Calculate metrics for a single pair of images, applying a scaling factor to the images.
    A mask is applied where the reference image values are bigger than 0.03.
    """
    try:
        masked_nac_img, masked_predicted_img, masked_reference_img = masked_SUV_img(nac_path, predicted_path, reference_path, nac_factor, mac_factor, mask_val)
        print("Masking successful.")
    except Exception as e:
        print(f"Error during masking: {e}")
    
    # Ensure no NaN values enter metric calculations
    valid_mask = (np.isfinite(masked_predicted_img) & np.isfinite(masked_reference_img) & (masked_reference_img > 0))
    if np.sum(valid_mask) == 0:
        return {key: np.nan for key in ["Mean Error (SUV)", "Mean Absolute Error (SUV)", "Relative Error (SUV%)", "Absolute Relative Error (SUV%)", "Root Mean Squared Error", "Peak Signal-to-Noise Ratio", "Structural Similarity Index"]}  # Default metric dictionary with NaN values

    # Apply the valid_mask to ensure no invalid data points are used in calculations
    filtered_predicted_img = masked_predicted_img[valid_mask]
    filtered_reference_img = masked_reference_img[valid_mask]

    peak = np.max([filtered_predicted_img.max(), filtered_reference_img.max()])
    metrics = {
        "Mean Error (SUV)": mean_error(filtered_predicted_img, filtered_reference_img),
        "Mean Absolure Error (SUV)": mean_absolute_error(filtered_predicted_img, filtered_reference_img),
        "Relative Error (SUV%)": relative_error(filtered_predicted_img, filtered_reference_img),
        "Absolure Relative Error (SUV%)": absolute_relative_error(filtered_predicted_img, filtered_reference_img),
        "Root Mean Squared Error": rmse(filtered_predicted_img, filtered_reference_img),
        "Peak Signal-to-Noise Ratio": psnr(filtered_predicted_img, filtered_reference_img, peak),
        "Structual Similarity Index": calculate_ssim(filtered_predicted_img, filtered_reference_img)
    }
    return metrics


def aggregate_metrics(metrics_list):
    """Aggregate metrics across all pairs and calculate mean and standard deviation."""
    aggregated_metrics = {key: [] for key in metrics_list[0]}
    for metrics in metrics_list:
        for key, value in metrics.items():
            aggregated_metrics[key].append(value)
    
    return {metric: (np.mean(values), np.std(values)) for metric, values in aggregated_metrics.items()}



def extract_suv_values(predicted_path, reference_path, scaling_factor, mask_val):
    """
    Extract SUV values for a single pair of images, applying a scaling factor to the images.
    A mask is applied where the reference image values are bigger than mask_val.
    """
    predicted_img = load_nifti_image(predicted_path) * scaling_factor
    reference_img = load_nifti_image(reference_path) * scaling_factor

    assert predicted_img.shape == reference_img.shape, "Loaded images must have the same shape."

    # Create mask from reference image where values are greater than mask_val
    mask = reference_img > mask_val
    
    # Apply the mask to both images
    masked_predicted_img = predicted_img[mask]
    masked_reference_img = reference_img[mask]
    
    assert masked_predicted_img.shape == masked_reference_img.shape, "Masked images must have the same shape."

    # Flatten the arrays
    predicted_flat = masked_predicted_img.ravel()
    reference_flat = masked_reference_img.ravel()

    # Final verification
    assert predicted_flat.shape == reference_flat.shape, "Flattened arrays must have the same shape."
    
    return predicted_flat, reference_flat



# import numpy as np
# import nibabel as nib
# from math import sqrt, log10
# from skimage.metrics import structural_similarity as ssim

# def mean_error(predicted, reference):
#     errors = predicted - reference
#     return np.mean(errors), np.std(errors)

# def mean_absolute_error(predicted, reference):
#     abs_errors = np.abs(predicted - reference)
#     return np.mean(abs_errors), np.std(abs_errors)

# def relative_error(predicted, reference):
#     re_errors = (predicted - reference) / reference
#     return np.mean(re_errors) * 100, np.std(re_errors) * 100

# def absolute_relative_error(predicted, reference):
#     are_errors = np.abs(predicted - reference) / reference
#     return np.mean(are_errors) * 100, np.std(are_errors) * 100

# def rmse(predicted, reference):
#     mse = np.mean((predicted - reference) ** 2)
#     return sqrt(mse), np.std((predicted - reference) ** 2) ** 0.5

# def psnr(predicted, reference, peak):
#     mse = np.mean((predicted - reference) ** 2)
#     psnr_value = 20 * log10(peak / sqrt(mse))
#     return psnr_value, 0

# def calculate_ssim(predicted, reference):
#     ssim_value = ssim(predicted, reference, data_range=reference.max() - reference.min())
#     return ssim_value, 0

# def load_nifti_image(path):
#     """Load a NIfTI image and return its data as a NumPy array."""
#     return nib.load(path).get_fdata()

# def calculate_metrics_for_pair(predicted_path, reference_path, scaling_factor, mask_val):
#     predicted_img = load_nifti_image(predicted_path) * scaling_factor
#     reference_img = load_nifti_image(reference_path) * scaling_factor
#     mask = reference_img > mask_val
#     masked_predicted_img = predicted_img[mask]
#     masked_reference_img = reference_img[mask]
#     peak = np.max([masked_predicted_img.max(), masked_reference_img.max()])
#     metrics = {
#         "mean_error": mean_error(masked_predicted_img, masked_reference_img),
#         "mean_absolute_error": mean_absolute_error(masked_predicted_img, masked_reference_img),
#         "relative_error": relative_error(masked_predicted_img, masked_reference_img),
#         "absolute_relative_error": absolute_relative_error(masked_predicted_img, masked_reference_img),
#         "rmse": rmse(masked_predicted_img, masked_reference_img),
#         # PSNR and SSIM are typically single-value metrics based on the overall image comparison
#         "psnr": psnr(masked_predicted_img, masked_reference_img, peak),
#         "ssim": calculate_ssim(masked_predicted_img, masked_reference_img)
#     }
#     # print(metrics)
#     return metrics


# def aggregate_metrics(metrics_list):
#     aggregated_metrics = {}
#     for metric in metrics_list[0].keys():  # Assuming all metrics are present in each entry
#         means = [pair_metrics[metric][0] for pair_metrics in metrics_list]
#         sds = [pair_metrics[metric][1] for pair_metrics in metrics_list]
        
#         # Calculate overall mean and pooled standard deviation
#         overall_mean = np.mean(means)
#         # For pooled standard deviation, considering simple averaging of individual variances
#         overall_sd = sqrt(np.mean([sd**2 for sd in sds]))
        
#         aggregated_metrics[metric] = (overall_mean, overall_sd)
#     return aggregated_metrics



# from quant import calculate_metrics_for_pair, aggregate_metrics
# # Calculate metrics for each pair and aggregate results
# all_metrics_dl3_18 = [calculate_metrics_for_pair(
#     pair['predicted'], pair['reference'],
#     scaling_factor=5, mask_val = 0.3)
#     for pair in all_pairs_dl3_18]

# metric_means_sds = aggregate_metrics(all_metrics_dl3_18)

# # Print aggregated metrics
# for metric, (mean, sd) in metric_means_sds.items():
#     print(f"{metric}: {mean:.2f} ± {sd:.4f}")