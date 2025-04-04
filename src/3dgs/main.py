import torch
import os
from tqdm import tqdm
from torch.multiprocessing import Process, Queue, set_start_method
from utils_new import GSProcessorWrapper, load_config
from data_preprocess import data_stream

from transformers import pipeline

if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    set_start_method('spawn', force=True)  # Ensure 'spawn' for CUDA compatibility
    torch.backends.cuda.preferred_linalg_library(backend="magma")

    # Initialize depth estimation pipeline
    pipe = pipeline(
        task="depth-estimation",
        model="xingyang1/Distill-Any-Depth-Large-hf",
        use_fast=True
    )

    # Load configuration and initialize processor
    config = load_config("../../config/config.yaml")
    processor = GSProcessorWrapper(config, "output", use_gui=False)

    # Define dataset parameters
    total_frames, batch_size = 180, 10
    data_dir = "../../dataset/red_sculpture"

    # Create a multiprocessing queue with a max size
    queue = Queue(maxsize=8)

    # Start data streaming in a separate process
    reader = Process(
        target=data_stream,
        args=(queue, data_dir, total_frames, batch_size, pipe)
    )
    reader.start()

    # Process frames with progress bar
    N = total_frames
    with tqdm(total=N, desc="Processing frames") as pbar:
        processed_frames = 0
        while True:
            t, data_packet, is_last = queue.get()  # Blocking call to get data
            num_frames = len(data_packet['viz_idx'])
            processed_frames += num_frames
            pbar.update(num_frames)

            if num_frames > 0:
                processor.process_data(data_packet)  # Process the batch
            pbar.set_description(
                f"Processing frame {processed_frames}/{total_frames} (batch {t + 1}/{N // batch_size + 1})"
            )

            if is_last:
                break

    # Finalize processing
    reader.join()
    poses = processor.finalize()
    print("Finished")