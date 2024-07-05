import torch
def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


path_to_checkpoint = "/usr/stud/tranv/storage/tranv/Research/OOD/BE5DW/nerfstudio/outputs/unnamed/splatfacto_segment_after_rgb/2024-07-03_125836/nerfstudio_models/step-000008500.ckpt"
checkpoint = torch.load(path_to_checkpoint)
dict_checkpoint = {}
dict_checkpoint["colors"] = SH2RGB(checkpoint["pipeline"]["_model.gauss_params.features_dc"])
dict_checkpoint["means"] = checkpoint["pipeline"]["_model.gauss_params.means"]
dict_checkpoint["scores"] = checkpoint["pipeline"]["_model.gauss_params.features_segmentation_small"]
torch.save(dict_checkpoint, "means_colors_scores_079a326597_fucking_bad.ckpt")
print("Hallo")