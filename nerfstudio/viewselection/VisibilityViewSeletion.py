from collections import defaultdict
from copy import deepcopy
import functools

import torch
from nerfstudio.viewselection.BaseViewSelection import BaseViewSelection
from nerfstudio.viewselection.BaseViewSelection import BaseViewSelection
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.splat_facto_segment import SplatfactoSegmentModel
from nerfstudio.engine.optimizers import Optimizers

from typing import Dict, List, Tuple

class VisibilityViewSeletion(BaseViewSelection):
    def __init__(self, dataset: List[Dict], optimizers: Optimizers, cameras: Cameras, min_visibility_gradient: float, model: SplatfactoSegmentModel, ratio_gaussians_covered: float, **kwargs):
        super().__init__(dataset=dataset, cameras=cameras, **kwargs)
        self.min_visibility_gradient = min_visibility_gradient
        self.model = model
        self.optimizers = optimizers
        self.ratio_gaussians_covered = ratio_gaussians_covered

    def get_visible_gaussians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        image_idx2visible_gaussians = []
        
        visible_gaussians_whole_scene = torch.full((self.model.num_points,), False, dtype=torch.bool)

        for i in range(len(self.dataset)):
            batch = self.dataset[i]
            outputs = self.model.get_outputs_uniform_segm_feature(self.cameras[i:i+1])
            loss_dict = self.model.get_loss_dict_segm(outputs, batch)
            loss = functools.reduce(torch.add, loss_dict.values())
            gradients = self.model.get_gradients_uniform_segm_feature()
            gradient_norms = torch.norm(gradients, dim=1)
            visible_gaussians = gradient_norms > self.min_visibility_gradient
            visible_gaussians_whole_scene = visible_gaussians_whole_scene | visible_gaussians
            image_idx2visible_gaussians.append(visible_gaussians)
            self.dataset[i]["visible_gaussians"] = visible_gaussians
            #Now the gradients need all to be zeroed
            self.model.zero_grad_uniform_segm_feature()
            needs_zero = [
                group for group in self.optimizers.parameters.keys()
            ]
            self.optimizers.zero_grad_some(needs_zero)

        return image_idx2visible_gaussians, visible_gaussians_whole_scene

    def get_visible_gaussians_for_id(self, classification_gaussians: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[int]]]:
        instance_idx2image_idx = self.ids_in_img() #self.dataset[i]["ids"]
        instance_id2visible_gaussians = [torch.full(self.model.num_points,False,dtype=torch.bool) for i in range(len(self.all_ids))]
        for i in range(len(self.dataset)):
            ids = self.dataset[i]["ids"] #these are the ids in this view
            visible_gaussians = self.dataset[i]["visible_gaussians"]
            for id in ids:
                visible_gaussians_id = visible_gaussians & (classification_gaussians == id)
                instance_id2visible_gaussians[id] = instance_id2visible_gaussians[id] | visible_gaussians_id
        instance_idx2image_idx = [list(instance_idx2image_idx[i]) for i in range(len(instance_idx2image_idx))]
        return instance_id2visible_gaussians, instance_idx2image_idx
    
    def select_view_for_id(self, visible_gaussians: torch.Tensor, views: List[int], image_idx2visible_gaussians: List[torch.Tensor]) -> Dict[int, float]:
        image_idx2visible_gaussians_instance_id = deepcopy(image_idx2visible_gaussians)
        image_idx2visible_gaussians_instance_id = [(image_idx2visible_gaussians_instance_id[i]&visible_gaussians, i) for i in range(len(image_idx2visible_gaussians_instance_id))]
        image_idx2visible_gaussians_instance_id.sort(lambda x: torch.sum(x[0]), reverse=True) # type: ignore
        visible_gaussians_instance_id = deepcopy(visible_gaussians)
        number_of_gaussians_instance_id = torch.sum(visible_gaussians_instance_id)
        def get_zero():
            return 0.0
        dict_id2view = defaultdict(get_zero)
        i=0
        while visible_gaussians_instance_id > number_of_gaussians_instance_id*self.ratio_gaussians_covered:
            #find view with most visible gaussians
            view = image_idx2visible_gaussians_instance_id[0][1]


    def select_views(self):
        #Plan: First get documentation for the model, which functions do I need?
        #-get_outputs_uniform_segm_feature(self, camera: Cameras)
        #-get_loss_dict_segm(self, outputs, batch, metrics_dict=None)
        #-get_gradients_uniform_segm_feature(self) return tensor of shape (number gaussians, number_of_instance_ids) NOTE: Implementation still reqinred
        #-classify_gaussians(self, gaussians) returns a tensor of shape (number gaussians) NOTE: Implementation still required
        """
        needs_zero = [
            group for group in self.optimizers.parameters.keys() if step % self.gradient_accumulation_steps[group] == 0
        ] #For which groups do we needto call zero_grad?
        self.optimizers.zero_grad_some(needs_zero)
        """
        #-zero_grad_uniform_segm_feature(self)

        #First, classify all Gaussians
        classification_gaussians = self.model.classify_gaussians()
        #get visible gaussians for every view
        image_idx2visible_gaussians, visible_gaussians_whole_scene = self.get_visible_gaussians()
        #Now, get all gaussians which are visible for evey id
        instance_id2visible_gaussians, instance_idx2image_idx = self.get_visible_gaussians_for_id(classification_gaussians)
        
        instance_ids2weight_dict = []
        for i in range(len(self.all_ids)):
            dict_view2weight = self.select_view_for_id(visible_gaussians=instance_id2visible_gaussians[i], views=instance_idx2image_idx[i], image_idx2visible_gaussians=image_idx2visible_gaussians)
            instance_ids2weight_dict.append(dict_view2weight)

        #Now calculate the masks
        self.calculate_masks(instance_ids2weight_dict)
