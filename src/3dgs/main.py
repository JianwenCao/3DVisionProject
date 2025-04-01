import torch
import os
from tqdm import tqdm
from torch.multiprocessing import Process, Queue
from utils_new import GSProcessorWrapper, load_config
from data_preprocess import data_stream
from DepthPrompting.model.ours.depth_prompt_main import depthprompting
from DepthPrompting.config import args as args_config
import torchvision.transforms as T
from PIL import Image


def setup_depth_model():
    """Initialize and configure the depth completion model."""
    args = args_config
    args.height, args.width = 256, 320  # Downsampled size for model
    args.max_depth = 30
    args.prop_kernel = 9
    args.prop_time = 18
    args.conf_prop = True
    args.pretrain = "./pretrained/Depthprompting_depthformer_kitti.tar"

    # Define transformations for RGB and depth
    t_rgb = T.Compose([
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Only normalization for tensor input
    ])
    t_dep = T.Compose([
        T.Resize(min(args.height, args.width), Image.BICUBIC),
        T.ToTensor()  # Convert PIL to tensor
    ])

    # Load and configure the depth model
    depth_model = depthprompting(args=args).cuda()
    checkpoint = torch.load(args.pretrain)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    new_state_dict = {n.replace("module.", ""): v for n, v in state_dict.items()}
    depth_model.load_state_dict(new_state_dict)
    depth_model.eval()

    return depth_model, t_rgb, t_dep

if __name__ == "__main__":
    # Set environment and multiprocessing settings
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.multiprocessing.set_start_method('spawn')
    torch.backends.cuda.preferred_linalg_library(backend="magma")

    # Load depth completion model and transformations
    depth_model, t_rgb, t_dep = setup_depth_model()

    # Load configuration and initialize processor
    config = load_config("../../config/config.yaml")
    processor = GSProcessorWrapper(config, "output", use_gui=True)

    # Define data parameters
    total_frames, frame_step, batch_size = 180, 1, 2
    data_dir = "../../dataset/red_sculpture"

    # Start data streaming process
    queue = Queue(maxsize=8)
    reader = Process(target=data_stream, args=(queue, data_dir, total_frames, frame_step,
                                               batch_size, depth_model, t_rgb, t_dep))
    reader.start()

    # Process frames with progress bar
    N = total_frames // frame_step
    with tqdm(total=N, desc="Processing frames") as pbar:
        processed_frames = 0
        while True:
            t, data_packet, is_last = queue.get()
            num_frames = len(data_packet['viz_idx'])
            processed_frames += num_frames
            pbar.update(num_frames)

            if num_frames > 0:
                processor.process_data(data_packet)
            pbar.set_description(
                f"Processing frame {processed_frames}/{total_frames} (batch {t + 1}/{N // batch_size + 1})")

            if is_last:
                break

    # Finalize processing
    reader.join()
    poses = processor.finalize()
    print("Finished")