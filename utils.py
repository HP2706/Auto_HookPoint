from torch import nn

def generate_expected_hookpoints(model, prefix=''):
    expected_hooks = []
    
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        
        # Add hook_point for the current module
        if not full_name.endswith('.hook_point'):
            expected_hooks.append(f"{full_name}.hook_point")        

        if isinstance(module, nn.ModuleList):
            for i, child in enumerate(module):
                expected_hooks.extend(generate_expected_hookpoints(child, f"{full_name}.{i}"))
        
        # Recursively process child modules
        expected_hooks.extend(generate_expected_hookpoints(module, full_name))
    
    return expected_hooks