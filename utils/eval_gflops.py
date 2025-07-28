import os

def eval_gflops_func(net, input_shape=None, args=None):
    print('')
    try:
        from torchinfo import summary
        from calflops import calculate_flops
    except:
        os.system("pip install torchinfo")
        os.system("pip install --upgrade calflops")
        os.system("pip install calflops-*-py3-none-any.whl")
        from torchinfo import summary
        from calflops import calculate_flops

    """ calflops """
    flops, macs, params = calculate_flops(model=net, input_shape=input_shape, args=args, output_as_string=True, output_precision=4)
    print(f"FLOPs: {flops},  MACs: {macs},   Params: {params}")
    return