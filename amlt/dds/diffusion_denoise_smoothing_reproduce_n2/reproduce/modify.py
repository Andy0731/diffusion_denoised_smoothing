with open('certify_0.25_mod', 'w') as outfile:
    cor_pre = 0
    i = 0
    with open('certify_0.25') as infile:
        for line in infile:
            i += 1
            if i == 1:
                outfile.write(line)
                continue
            print(line)
            idx = int(line.split()[0])
            correct = int(line.split()[-2])
            l = line.split()
            if (idx < 16 and correct == 1) or (idx >= 16 and correct > cor_pre):
                cor_mod = 1
            else:
                cor_mod = 0
            l[-2] = str(cor_mod)
            ls = '\t'.join(l)
            outfile.write(ls + '\n')
            print(ls)
            cor_pre = correct