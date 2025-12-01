
def debug_optimizer(optimizer_params):
    print("--- Examining Optimizer Parameters ---")

    for i, param_group in enumerate(optimizer_params):
        # Get the learning rate for the group
        lr = param_group['lr']

        # Get the list of parameter tensors for the group
        params = param_group['params']

        # Calculate the total number of parameters in this group
        num_params = sum(p.numel() for p in params)

        print(f"Group {i}:")
        print(f"  - Learning Rate: {lr}")
        print(f"  - Number of Tensors: {len(params)}")
        print(f"  - Total Parameters: {num_params:,}")  # Formats with commas
        print(
            f"  - Shapes of first 3 tensors: {[p.shape for p in params[:3]]}")

    print("-" * 35)
