import math


def dot_product(a,b):
    return sum(x*y for x,y in zip(a,b))


def magnitude(vector):
    return math.sqrt(sum(value**2 for value in vector))

def cosine_similarity(a,b):
    if len(a) != len(b):
        raise ValueError(
            "Vectors must have same dimensions"
        )
    magnitude_a = magnitude(a)
    magnitude_b = magnitude(b)

    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError("Cosine Similarity is undefined for zero vectors")
    return dot_product(a,b)/ (magnitude_a* magnitude_b)

if __name__ == "__main__":
    a = [1.0, 0.0]
    b = [0.9, 0.1]
    c = [0.0, 1.0]

    print("a vs a:", cosine_similarity(a,a))
    print("a vs b:", cosine_similarity(a,b))
    print("a vs c:", cosine_similarity(a,c))
    