import os
import sys

def writeLatexTables(results, save_filename):
    methods = list(results.keys())
    samples = results[methods[0]]['samples']

    n_samples = len(samples)
    n_methods = len(methods)

    with open(save_filename, 'w') as fp:
        writeLine(fp, "\\begin{table}[h!]")
        writeLine(fp, "\centering")
        writeLine(fp, "\\begin{{tabular}}{{ {} }}".format((n_samples + 1) * '|l' + '|'))
        writeLine(fp, "    \hline")
        writeLine(fp, f"    \multirow{{2}}{{*}}{{Method}} & \multicolumn{{{n_samples}}}{{c|}}{{Relative error ($\%$)}}   \\\\ \cline{{2-{n_samples+1}}}")
        
        samples_string = ''
        for sample in samples:
            samples_string += f'& N = {sample}'
        writeLine(fp, f"      {samples_string}\\\\ \cline{{2-{n_samples+1}}}")
        writeLine(fp, "    \hline")
        for method in methods:
            curr_results = ''
            for err in results[method]['rel_error']:
                curr_results += ' & ' + '{:0.2f}'.format(err*100) 
            writeLine(fp, f"    \\text{{{method}}} {curr_results}  \\\\ \cline{{1-{n_samples+1}}}")
        writeLine(fp, "  \end{tabular}")
        writeLine(fp, "  \caption{INSERT CAPTION}")
        writeLine(fp, "  \label{INSERT LABEL}")
        writeLine(fp, "\end{table}")

        
def writeLine(fp, line):
    fp.write(line + '\n')
