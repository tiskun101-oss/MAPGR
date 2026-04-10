import argparse

from datasets.base import DunhangDataset, DhMuralDataset


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def dataset(folder,
            image_size,
            exts=['jpg', 'jpeg', 'png', 'tiff'],
            augment_flip=False,
            convert_image_to=None,
            equalizeHist=False,
            crop_patch=True,
            sample=False,
            split_mode=None,
            alpha_power=1.0,
            alpha_hard_threshold=None,
            mask_binary_threshold=0.5,
            mask_select_mode="random",
            is_dunhuang=True,
            agent_meta=False,
            prompt_cache=None,
            prompt_cache_path=None):
    
    if is_dunhuang:
        return DunhangDataset(folder,
                        image_size,
                        exts=exts,
                        augment_flip=augment_flip,
                        convert_image_to=convert_image_to,
                        equalizeHist=equalizeHist,
                        crop_patch=crop_patch,
                        sample=sample,
                        split_mode=split_mode,
                        alpha_power=alpha_power,
                        alpha_hard_threshold=alpha_hard_threshold,
                        mask_binary_threshold=mask_binary_threshold,
                        mask_select_mode=mask_select_mode,
                        agent_meta=agent_meta,
                        prompt_cache=prompt_cache,
                        prompt_cache_path=prompt_cache_path)
    else:
        return MuralDataset(folder,
                        image_size,
                        exts=exts,
                        augment_flip=augment_flip,
                        convert_image_to=convert_image_to,
                        equalizeHist=equalizeHist,
                        crop_patch=crop_patch,
                        sample=sample,
                        split_mode=split_mode,
                        alpha_power=alpha_power,
                        alpha_hard_threshold=alpha_hard_threshold,
                        mask_binary_threshold=mask_binary_threshold,
                        mask_select_mode=mask_select_mode,
                        agent_meta=agent_meta,
                        prompt_cache=prompt_cache,
                        prompt_cache_path=prompt_cache_path)
