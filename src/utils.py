

def save_run_config(args, path):
    with open(path, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")