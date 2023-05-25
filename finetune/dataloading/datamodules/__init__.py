from .rgz import RGZ_DataModule_Finetune, MiraBest_FITS_DataModule_Finetune

finetune_datasets = {
    # "stl10": STL10_DataModule,
    # "imagenette": Imagenette_DataModule,
    "rgz": RGZ_DataModule_Finetune,
    "mb_fits": MiraBest_FITS_DataModule_Finetune,
}
