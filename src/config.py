def get_config():
    return {
        'data_path' : 'AIProjectFabricComp\\data\\raw_data\\fabrics',
        'data_folder' : 'AIProjectFabricComp\\data',
        'training' : True,
        'batch_size' : 16,
        'feature_dim' : 2048,
        'higher_dim' : 4096,
        'num_heads' : 4,
        'num_clothes' : 14,
        'num_fabrics' : 28,
        'dropout' : 0.05,
        'lr' : 1e-3,
        
    }
    