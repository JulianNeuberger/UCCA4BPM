import os

for filename in [f for f in os.listdir('.') if f.endswith('.txt')]:
    filename = filename.replace('.txt', '')

    with open(f'{filename}.txt') as t:
        t = t.read()
        with open(f'{filename}.ann') as a:
            for l in a:
                l = l.split('\t')
                if l[0].startswith('T'):
                    clazz, token_s, token_e = l[1].split(' ')
                    if clazz == 'Entity' and 'claim' in t[int(token_s): int(token_e)]:
                        epsilon = 10
                        print(f'"{t[max(int(token_s) - epsilon, 0): int(token_e) + epsilon]}" should be "{l[2][:-1]}" ({clazz})')
