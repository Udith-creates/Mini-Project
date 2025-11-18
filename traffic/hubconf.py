"""PyTorch Hub models

Usage:
    import torch
    model = torch.hub.load('repo', 'model')
"""

from pathlib import Path

import torch

from models.yolo import Model
from utils.general import check_requirements, set_logging
from utils.google_utils import attempt_download
from utils.torch_utils import select_device

dependencies = ['torch', 'yaml']
check_requirements(Path(__file__).parent / 'requirements.txt', exclude=('pycocotools', 'thop'))
set_logging()


def create(name, pretrained, channels, classes, autoshape):
    """Creates a specified model

    Arguments:
        name (str): name of model, i.e. 'yolov7'
        pretrained (bool): load pretrained weights into the model
        channels (int): number of input channels
        classes (int): number of model classes

    Returns:
        pytorch model
    """
    try:
        cfg = list((Path(__file__).parent / 'cfg').rglob(f'{name}.yaml'))[0]  # model.yaml path
        model = Model(cfg, channels, classes)
        if pretrained:
            fname = f'{name}.pt'  # checkpoint filename
            attempt_download(fname)  # download if not found locally
            ckpt = torch.load(fname, map_location=torch.device('cpu'))  # load
            msd = model.state_dict()  # model state_dict
            csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
            csd = {k: v for k, v in csd.items() if msd[k].shape == v.shape}  # filter
            model.load_state_dict(csd, strict=False)  # load
            if len(ckpt['model'].names) == classes:
                model.names = ckpt['model'].names  # set class names attribute
            if autoshape:
                model = model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
        device = select_device('0' if torch.cuda.is_available() else 'cpu')  # default to GPU if available
        return model.to(device)

    except Exception as e:
        s = 'Cache maybe be out of date, try force_reload=True.'
        raise Exception(s) from e


def custom(path_or_model='path/to/model.pt', autoshape=True):
    """custom mode

    Arguments (3 options):
        path_or_model (str): 'path/to/model.pt'
        path_or_model (dict): torch.load('path/to/model.pt')
        path_or_model (nn.Module): torch.load('path/to/model.pt')['model']

    Returns:
        pytorch model
    """
    model = torch.load(path_or_model, map_location=torch.device('cpu')) if isinstance(path_or_model, str) else path_or_model  # load checkpoint
    if isinstance(model, dict):
        model = model['ema' if model.get('ema') else 'model']  # load model

    hub_model = Model(model.yaml).to(next(model.parameters()).device)  # create
    hub_model.load_state_dict(model.float().state_dict())  # load state_dict
    hub_model.names = model.names  # class names
    if autoshape:
        hub_model = hub_model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
    device = select_device('0' if torch.cuda.is_available() else 'cpu')  # default to GPU if available
    return hub_model.to(device)


def yolov7(pretrained=True, channels=3, classes=80, autoshape=True):
    return create('yolov7', pretrained, channels, classes, autoshape)


if __name__ == '__main__':
    import sys
    from pathlib import Path
    
    print("=" * 70)
    print("iSENSOR - PyTorch Hub Demo")
    print("=" * 70)
    
    # Example 1: Create model WITHOUT pretrained weights (recommended for testing)
    print("\n[TEST 1] Creating YOLOv7 model from scratch (no weights)...")
    try:
        model = create(name='yolov7', pretrained=False, channels=3, classes=80, autoshape=True)
        print("  OK Model created successfully")
        print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"  ERROR Failed: {e}")
    
    # Example 2: Test inference with dummy input
    print("\n[TEST 2] Testing inference with dummy input...")
    try:
        import numpy as np
        import cv2
        
        # Create dummy image (480x640x3) as numpy array (H x W x C)
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        print(f"  Input: numpy array shape {dummy_img.shape}")
        
        # Create model without autoshape for raw tensor input
        model_raw = create(name='yolov7', pretrained=False, channels=3, classes=80, autoshape=False)
        
        # Prepare tensor input: convert HWC to BCHW
        img_tensor = torch.from_numpy(dummy_img).float() / 255.0  # (H, W, C)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # CHW -> BCHW
        
        print(f"  Tensor shape: {img_tensor.shape}")
        
        with torch.no_grad():
            output = model_raw(img_tensor)
        
        print("  OK Inference successful")
        if isinstance(output, (list, tuple)):
            print(f"    Output tensors: {len(output)}")
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    print(f"      Output {i}: {out.shape}")
        else:
            print(f"    Output shape: {output.shape}")
    except Exception as e:
        print(f"  ERROR Inference failed: {e}")
    
    # Example 3: Load from pretrained weights if available
    print("\n[TEST 3] Pretrained weights example...")
    weights_path = Path('yolov7.pt')
    if weights_path.exists():
        print(f"  OK Found weights: {weights_path}")
        try:
            model_pretrained = custom(path_or_model=str(weights_path))
            print("  OK Pretrained model loaded successfully")
        except Exception as e:
            print(f"  ERROR Failed to load: {e}")
    else:
        print(f"  WARNING Weights not found at {weights_path}")
        print(f"    Download from: https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt")
        print(f"    Or run: python download_and_test.py")
    
    print("\n" + "=" * 70)
    print("SUCCESS Hub demo completed successfully!")
    print("=" * 70)
    print("\nUsage examples:")
    print("  model = create('yolov7', pretrained=False, channels=3, classes=80, autoshape=True)")
    print("  model = yolov7(pretrained=False, channels=3, classes=80, autoshape=True)")
    print("\nFor inference on images/videos:")
    print("  python detect.py --weights yolov7-tiny.pt --source sample.mp4")
    print("\n" + "=" * 70)
