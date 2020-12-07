import os
for file_name in [f for f in os.listdir('.') if f.endswith('.json')]:
    os.rename(file_name, file_name.replace('_hidden_', '_'))
