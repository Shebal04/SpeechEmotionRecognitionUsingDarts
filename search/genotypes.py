# search/genotypes.py

from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

# Example of predefined genotype (a representation of a specific architecture)
PRIMITIVES = [
    'none', 'identity', 'avg_pool', 'max_pool', 'conv_3x3', 'conv_5x5', 'lstm', 'gru'
]

# Example genotype
example_genotype = Genotype(
    normal=[
        ('conv_3x3', 0), ('conv_5x5', 1), ('max_pool', 2), ('lstm', 3), 
        ('gru', 4), ('identity', 5), ('conv_5x5', 6), ('avg_pool', 7)
    ],
    normal_concat=[2, 4, 6],
    reduce=[
        ('max_pool', 0), ('avg_pool', 1), ('conv_3x3', 2), ('identity', 3),
        ('lstm', 4), ('gru', 5), ('conv_5x5', 6), ('none', 7)
    ],
    reduce_concat=[2, 4, 6]
)

def parse_genotype(string):
    """
    Parse a genotype string into a Genotype object.
    """
    parts = string.split('normal:')
    normal_part = parts[1].split('normal_concat:')[0].strip()
    normal_concat_part = parts[1].split('normal_concat:')[1].split('reduce:')[0].strip()
    reduce_part = parts[1].split('reduce:')[1].split('reduce_concat:')[0].strip()
    reduce_concat_part = parts[1].split('reduce_concat:')[1].strip()

    # Convert string representations into Python objects
    normal = eval(normal_part)  # Convert string to actual list
    normal_concat = eval(normal_concat_part)
    reduce = eval(reduce_part)
    reduce_concat = eval(reduce_concat_part)

    # Return the parsed Genotype
    return Genotype(normal, normal_concat, reduce, reduce_concat)

def genotype_to_str(genotype):
    """
    Convert a genotype object into a string representation.
    """
    normal_str = str(genotype.normal)
    normal_concat_str = str(genotype.normal_concat)
    reduce_str = str(genotype.reduce)
    reduce_concat_str = str(genotype.reduce_concat)

    return f"normal: {normal_str} normal_concat: {normal_concat_str} reduce: {reduce_str} reduce_concat: {reduce_concat_str}"

def save_genotype_to_file(genotype, filename='genotype.txt'):
    """
    Save the genotype as a string in a file.
    """
    with open(filename, 'w') as f:
        f.write(genotype_to_str(genotype))

def load_genotype_from_file(filename='genotype.txt'):
    """
    Load a genotype from a file.
    """
    with open(filename, 'r') as f:
        genotype_str = f.read()
    return parse_genotype(genotype_str)

