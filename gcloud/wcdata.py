with open('wcdata', 'w') as fout:
    for i in range(10):
        fout.write("%i\tEllie\n" % i)
    for i in range(15):
        fout.write("%i\tTatoshka\n" % i)
