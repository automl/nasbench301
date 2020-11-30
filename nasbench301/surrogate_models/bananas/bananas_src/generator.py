import sys
import os
from hpbandster.optimizers.config_generators.darts_src.genotypes import Genotype

def get_op(node_idx, prev_node_idx, op_idx, normal=True):
    primitives = prims['primitives_normal' if normal else 'primitives_reduct']

    node_to_in_edges = {
        0: (0, 2),
        1: (2, 5),
        2: (5, 9),
        3: (9, 14)
    }

    in_edges = node_to_in_edges[node_idx]
    op = primitives[in_edges[0]: in_edges[1]][prev_node_idx][op_idx]
    return op

def extract_genes(log_file, name='darts'):
    with open(log_file) as f:
        if name == 'random_nas':
            lines = [eval(x[24:-1]) for x in f.readlines() if x[24:26] == "(["][:50]
        elif name == 'gdas':
            lines = [eval(x[x.find('genotype') + len('genotype = '): -1])
                     for x in f.readlines() if 'genotype' in x]
            l = []
            for arch in lines:
                normal=[(x[0], x[1]) for x in arch.normal]
                reduce=[(x[0], x[1]) for x in arch.reduce]
                gene = Genotype(normal=normal, normal_concat=range(2, 6),
                                reduce=reduce, reduce_concat=range(2, 6))
                l.append(gene)
            lines = l
        else:
            lines = [eval(x[x.find('genotype') + len('genotype = '): -1])
                     for x in f.readlines() if 'genotype' in x]
    return lines


def main(seed, name='darts', space='s1'):
    genes = extract_genes('logs/'+name+'/'+space+'/log_%d.txt'%seed, name)

    for k, g in enumerate(genes):
        print('INDEX: %d'%k)
        if name == 'random_nas':
            darts_arch = [[], []]
            for i, (cell, normal) in enumerate(zip(g, [True, False])):
                for j, n in enumerate(cell):
                    darts_arch[i].append((get_op(j//2, n[0], n[1], normal),
                                          n[0]))

            arch_str = 'Genotype(normal=%s, normal_concat=range(2, 6), reduce=%s, reduce_concat=range(2, 6))' % (str(darts_arch[0]), str(darts_arch[1]))
        else:
            arch_str = str(g)

        print(arch_str)
        if not os.path.exists('genes/{}'.format(space)):
            os.makedirs('genes/{}'.format(space))
        with open('genes/{}/{}_{}.txt'.format(space, name, seed), 'a') as f:
            f.write('DARTS_'+str(k)+' = ' + arch_str)
            f.write('\n')


if __name__=="__main__":
    name = sys.argv[1]
    space = sys.argv[2]
    a = int(sys.argv[3])
    b = int(sys.argv[4])
    for seed in range(a, b):
        main(seed, name, space)

