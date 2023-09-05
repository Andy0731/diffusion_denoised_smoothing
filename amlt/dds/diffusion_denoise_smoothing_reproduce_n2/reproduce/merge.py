def merge_ctf_files(ctf_filename):
    with open(ctf_filename, 'w') as outfile:
        for i in range(16):
            fname = ctf_filename + '_rank' + str(i)
            with open(fname) as infile:
                outfile.write(infile.read())
    return

merge_ctf_files('certify_0.25')