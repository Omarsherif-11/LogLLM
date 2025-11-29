import os
import sys
import time
import random
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from torch import nn, optim
from cp_model import LogLLM
from torch.utils.data import DataLoader
from customDataset import CustomDataset, CustomCollator, BalancedSampler

# --- CONFIGURATION ---
MAX_DURATION = 27 * 60  # 27 Minutes (in seconds). Leaves 3 mins for saving.
START_TIME = time.time()

n_epochs_1 = 1
n_epochs_2_1 = 1
n_epochs_2_2 = 1
n_epochs_3 = 2
dataset_name = 'bad_windows'
batch_size = 16
micro_batch_size = 4
gradient_accumulation_steps = batch_size // micro_batch_size

lr_1 = 5e-4
lr_2_1 = 5e-4
lr_2_2 = 5e-5
lr_3 = 5e-5
max_content_len = 100
max_seq_len = 128

data_path = r'/pfs/data6/home/hu/hu_hu/hu_abdeom01/logllm_work/datasets/{}/prepared/train.csv'.format(dataset_name)
min_less_portion = 0.3
Bert_path = r"/pfs/data6/home/hu/hu_hu/hu_abdeom01/logllm_work/bert-base-uncased"
Llama_path = r"/pfs/data6/home/hu/hu_hu/hu_abdeom01/logllm_work/Meta-Llama-3-8B"

ROOT_DIR = Path(__file__).parent
# Directory for final exported model
ft_path = os.path.join(ROOT_DIR, r"ft_model_{}".format(dataset_name))
# Directory for intermediate training checkpoints
checkpoint_dir = os.path.join(ROOT_DIR, r"checkpoints_{}".format(dataset_name))
checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")

device = torch.device("cuda:0")

# Ensure reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def save_training_state(model, optimizer, scheduler, phase_name, epoch, step, loss_acc, dataset_rng_states):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Save Model Weights (Separated because they are heavy and handled by model class)
    model.save_checkpoint(checkpoint_dir)
    
    # Save Training Meta-data
    state = {
        'phase': phase_name,
        'epoch': epoch,
        'step': step,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state(),
        'np_rng_state': np.random.get_state(),
        'py_rng_state': random.getstate(),
        'loss_acc': loss_acc # Preserve logging counters
    }
    torch.save(state, checkpoint_path)
    print(f"-> Checkpoint saved at Phase: {phase_name}, Epoch: {epoch}, Step: {step}")

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    print(f"all params: {all_model_params}, trainable: {trainable_model_params}")
    return trainable_model_params

def check_time_limit_and_save(model, optimizer, scheduler, phase, epoch, step, loss_acc):
    elapsed = time.time() - START_TIME
    if elapsed > MAX_DURATION:
        print(f"!! Time limit reached ({elapsed/60:.2f} mins). Saving and Exiting...")
        save_training_state(model, optimizer, scheduler, phase, epoch, step, loss_acc, None)
        sys.exit(0) # Exit cleanly

def trainModel(model, dataloader, gradient_accumulation_steps, n_epochs, lr, phase_name):
    criterion = nn.CrossEntropyLoss(reduction='mean')

    print_number_of_trainable_model_parameters(model)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)
    
    start_epoch = 0
    start_step = 0
    resume_mode = False
    
    total_acc, total_acc_count, total_count, train_loss = 0, 0, 0, 0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if checkpoint['phase'] == phase_name:
            print(f"-> Resuming {phase_name} from Epoch {checkpoint['epoch']}, Step {checkpoint['step']}")
            
            model.load_checkpoint(checkpoint_dir)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # --- FIX IS HERE ---
            torch.set_rng_state(checkpoint['rng_state'].cpu()) # Force CPU
            # -------------------
            
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'].cpu())
            np.random.set_state(checkpoint['np_rng_state'])
            random.setstate(checkpoint['py_rng_state'])
            
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step']
            
            if 'loss_acc' in checkpoint:
                total_acc, total_acc_count, total_count, train_loss = checkpoint['loss_acc']
            
            resume_mode = True
        else:
            print(f"-> Found checkpoint for {checkpoint['phase']}, but currently starting {phase_name}. Ignoring.")
            if os.path.exists(os.path.join(checkpoint_dir, 'model_weights.pt')):
                 print(f"-> Loading weights from previous phase...")
                 model.load_checkpoint(checkpoint_dir)

    # Tokens for accuracy calculation
    normal_tokens = model.Llama_tokenizer('The sequence is normal.')['input_ids']
    anomalous_tokens = model.Llama_tokenizer('The sequence is anomalous.')['input_ids']
    special_normal_tokens = set(normal_tokens) - set(anomalous_tokens)
    special_anomalous_tokens = set(anomalous_tokens) - set(normal_tokens)

    total_steps = n_epochs * len(dataloader)
    scheduler_step = max(int(total_steps / 10), 1)
    print(f'scheduler_step: {scheduler_step}')

    global_step_counter = 0

    for epoch in range(start_epoch, int(n_epochs)):
        pbar = tqdm(dataloader, desc=f'[{phase_name}] Epoch {epoch}/{n_epochs}')
        
        for i_th, batch_i in enumerate(pbar):
            
            # --- SKIP LOGIC FOR RESUMPTION ---
            # If we are in the resume epoch, and haven't reached the saved step yet
            if resume_mode and epoch == start_epoch and i_th < start_step:
                # We need to increment global counters to keep scheduler sync logic correct implicitly?
                # Actually, scheduler state is loaded, so we just need to increment loop counter
                if (i_th + 1) % gradient_accumulation_steps == 0:
                   pass 
                # Purely skip
                continue
            
            # Once we pass the skip point, disable resume mode flag so we don't skip in next epochs
            if resume_mode and epoch == start_epoch and i_th >= start_step:
                resume_mode = False
            
            global_step_counter += 1

            inputs= batch_i['inputs']
            seq_positions= batch_i['seq_positions']
            labels = batch_i['labels']

            inputs = inputs.to(device)
            seq_positions = seq_positions

            outputs, targets = model.train_helper(inputs, seq_positions, labels)

            loss = criterion(outputs, targets)
            loss = loss / gradient_accumulation_steps

            loss.backward()

            if ((i_th + 1) % gradient_accumulation_steps == 0) or ((i_th + 1) == len(dataloader)):
                optimizer.step() 
                optimizer.zero_grad() 

            # Accuracy calculation
            acc_mask = torch.zeros_like(targets,device=device).bool()
            for token in special_normal_tokens.union(special_anomalous_tokens):
                acc_mask[targets == token] = True

            total_acc += (outputs.argmax(1)[acc_mask] == targets[acc_mask]).sum().item()
            total_acc_count += acc_mask.sum()
            train_loss += loss.item() * gradient_accumulation_steps * targets.size(0)
            total_count += targets.size(0)

            # Scheduler step
            # Note: The original code used a separate 'steps' counter that accumulated across epochs. 
            # We reconstruct it approximately.
            current_accum_steps = (epoch * len(dataloader) + i_th) + 1
            if current_accum_steps % scheduler_step == 0:
                scheduler.step()
            
            pbar.set_postfix(lr=scheduler.get_last_lr()[0], loss = loss.item() * gradient_accumulation_steps)

            if current_accum_steps % 10000 == 0:
                train_loss_epoch = train_loss / total_count if total_count > 0 else 0
                train_acc_epoch = total_acc / total_acc_count if total_acc_count > 0 else 0
                print(f"[Epoch {epoch + 1}/{n_epochs}] [loss: {train_loss_epoch:3f}] [acc: {train_acc_epoch:3f}]")
                # Reset counters
                total_acc, total_acc_count, total_count, train_loss = 0, 0, 0, 0

            # --- TIME CHECK ---
            # We check every step (or every N steps)
            if (i_th + 1) % 10 == 0: # Check every 10 batches to save overhead
                loss_stats = (total_acc, total_acc_count, total_count, train_loss)
                # Save 'i_th + 1' so when we resume, we start at the next batch
                check_time_limit_and_save(model, optimizer, scheduler, phase_name, epoch, i_th + 1, loss_stats)

        # End of Epoch Stats
        if total_count > 0:
            train_loss_epoch = train_loss / total_count
            train_acc_epoch = total_acc / total_acc_count
            print(f"End Epoch {epoch + 1}: loss: {train_loss_epoch:3f} acc: {train_acc_epoch:3f}")

    # End of Phase
    print(f"Phase {phase_name} Completed.")
    # Mark phase as done in checkpoint logic by effectively deleting the checkpoint for this phase 
    # or updating it to the next phase immediately.
    # Here, we will delete the checkpoint file so the main loop enters the next phase cleanly.
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    # Save weights only (for the next phase to load)
    model.save_checkpoint(checkpoint_dir)

if __name__ == '__main__':
    dataset = CustomDataset(data_path, drop_duplicates=False)

    model = LogLLM(Bert_path, Llama_path, device=device, max_content_len=max_content_len, max_seq_len=max_seq_len)
    tokenizer = model.Bert_tokenizer
    collator = CustomCollator(tokenizer, max_seq_len=max_seq_len, max_content_len=max_content_len)

    # Define the Phases
    # Format: (PhaseName, ModeFunc, Epochs, LR, UseMaxSamples)
    phases = [
        ("phase_1", model.set_train_only_Llama, n_epochs_1, lr_1, True),
        ("phase_2_1", model.set_train_only_projector, n_epochs_2_1, lr_2_1, False),
        ("phase_2_2", model.set_train_projectorAndBert, n_epochs_2_2, lr_2_2, False),
        ("phase_3", model.set_finetuning_all, n_epochs_3, lr_3, False)
    ]

    # Determine Start Phase
    start_index = 0
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu') # Load just to check phase
        last_phase = ckpt.get('phase')
        for i, (name, _, _, _, _) in enumerate(phases):
            if name == last_phase:
                start_index = i
                print(f"Found checkpoint in progress for {name}. Restarting loop at index {i}.")
                break
    else:
        # If no checkpoint exists, but we have weights, it means we completed a phase previously 
        # but crashed/stopped *between* phases. 
        # Since we delete checkpoint.pt on completion, we rely on the logic below to flow sequentially.
        # But to be safe on restarts between phases, we could use a separate "progress.txt" file.
        # For now, we assume if checkpoint.pt is gone, we check which phase we "look" like we are in? 
        # No, strict ordering is safer.
        
        # Improvement: Check for a "completed_phases.txt"
        done_file = os.path.join(checkpoint_dir, "completed_phases.txt")
        if os.path.exists(done_file):
            with open(done_file, 'r') as f:
                completed = f.read().strip().split('\n')
            
            # Find the first phase NOT in completed
            for i, (name, _, _, _, _) in enumerate(phases):
                if name not in completed:
                    start_index = i
                    break
                else:
                    # If phase is done, ensure we load the weights resulting from it!
                    if os.path.exists(os.path.join(checkpoint_dir, 'model_weights.pt')):
                        print(f"Skipping {name} (Already done).")
                        start_index = i + 1 

    # Load most recent weights if we are starting fresh or moving to next phase
    if os.path.exists(os.path.join(checkpoint_dir, 'model_weights.pt')):
        print("Loading latest model weights...")
        model.load_checkpoint(checkpoint_dir)

    for i in range(start_index, len(phases)):
        p_name, p_mode_func, p_epochs, p_lr, p_max_samples = phases[i]
        
        print(f"\n{'='*10} STARTING {p_name} {'='*10}")
        
        # Apply Mode
        p_mode_func()
        
        # Setup Dataloader
        if p_max_samples:
             sampler = BalancedSampler(dataset, target_ratio=min_less_portion, max_samples=1000)
             print("Using BalancedSampler with max_samples=1000")
        else:
             sampler = BalancedSampler(dataset, target_ratio=min_less_portion)
             print("Using standard BalancedSampler")

        dataloader = DataLoader(
            dataset,
            batch_size=micro_batch_size,
            num_workers=2,
            sampler=sampler,
            collate_fn=collator,
            drop_last=True
        )

        # Run Training
        trainModel(model, dataloader, gradient_accumulation_steps, p_epochs, p_lr, p_name)
        
        # Mark phase complete
        with open(os.path.join(checkpoint_dir, "completed_phases.txt"), 'a') as f:
            f.write(p_name + "\n")
        
        # Explicit garbage collection
        del dataloader
        torch.cuda.empty_cache()

    # Final Save
    model.save_ft_model(ft_path)
    print("Training Pipeline Completed Successfully.")

    with open("training_completed.done", "w") as f:
        f.write("done")