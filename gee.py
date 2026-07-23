import os

# Add to .bashrc file
bashrc_path = os.path.expanduser('~/.bashrc')
env_line = 'export EARTHENGINE_PROJECT=526780702263\n'

# Check if it's already there
with open(bashrc_path, 'r') as f:
    content = f.read()

if 'EARTHENGINE_PROJECT' not in content:
    with open(bashrc_path, 'a') as f:
        f.write(env_line)
    print('✓ Added EARTHENGINE_PROJECT to ~/.bashrc')
else:
    print('✓ EARTHENGINE_PROJECT already in ~/.bashrc')

print('\nRun this in terminal to apply changes:')
print('source ~/.bashrc')