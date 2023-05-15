from amc_parser import parse_amc
import numpy as np

def flatten(l):
    """
    Get flattened list from list of lists
    :param l: list of lists

    :return flattened list
    """
    return [item for sublist in l for item in sublist]

def amc2txt(sequence_path, save_path):
    """
    Convert *.amc file to *.txt
    Rows of the *.txt file are the time frames
    Columns of the *.txt file are the DOFs of the skeleton
    :param sequence_path: Path to the *.amc file
    :param save_path: Path where the *.txt file will be saved
    """
    amc_path = sequence_path
    motions = parse_amc(amc_path)

    DOF = sum(len(motions[0][key]) for key in motions[0].keys())

    sequence = np.zeros((len(motions), DOF))
    
    for frame in range(len(motions)):
        sequence[frame, :] = flatten([motions[frame][key] for key in motions[frame].keys()])

    np.savetxt(save_path, sequence)
    return sequence


def txt2amc(sequence_path, save_path):
    """
    Save motion sequence found in *.txt file to *.amc file for visualization 
    Rows of the *.txt file are the time frames
    Columns of the *.txt file are the DOFs of the skeleton
    :param sequence_path: Path to the *.txt file
    :param save_path: Path where the *.amc file will be saved
    """

    D = np.loadtxt(sequence_path)

    frames, DOF = D.shape

    with open(save_path, 'w') as fid:
        # Write the header information
        fid.write(':FULLY-SPECIFIED\n')
        fid.write(f':DEGREES\n')

        for frame in range(frames):
            fid.write('%d\n' % frame)
            
            fid.write('root %f %f %f %f %f %f\n' % tuple(D[frame, 0:6]))
            fid.write('lowerback %f %f %f\n' % tuple(D[frame, 6:9]))
            fid.write('upperback %f %f %f\n' % tuple(D[frame, 9:12]))
            fid.write('thorax %f %f %f\n' % tuple(D[frame, 12:15]))
            fid.write('lowerneck %f %f %f\n' % tuple(D[frame, 15:18]))
            fid.write('upperneck %f %f %f\n' % tuple(D[frame, 18:21]))
            fid.write('head %f %f %f\n' % tuple(D[frame, 21:24]))
            fid.write('rclavicle %f %f\n' % tuple(D[frame, 24:26]))
            fid.write('rhumerus %f %f %f\n' % tuple(D[frame, 26:29]))
            fid.write('rradius %f\n' % D[frame, 29])
            fid.write('rwrist %f\n' % D[frame, 30])
            fid.write('rhand %f %f\n' % tuple(D[frame, 31:33]))
            fid.write('rfingers %f\n' % D[frame, 33])
            fid.write('rthumb %f %f\n' % tuple(D[frame, 34:36]))
            fid.write('lclavicle %f %f\n' % tuple(D[frame, 36:38]))
            fid.write('lhumerus %f %f %f\n' % tuple(D[frame, 38:41]))
            fid.write('lradius %f\n' % D[frame, 41])
            fid.write('lwrist %f\n' % D[frame, 42])
            fid.write('lhand %f %f\n' % tuple(D[frame, 43:45]))
            fid.write('lfingers %f\n' % D[frame, 45])
            fid.write('lthumb %f %f\n' % tuple(D[frame, 46:48]))
            fid.write('rfemur %f %f %f\n' % tuple(D[frame, 48:51]))
            fid.write('rtibia %f\n' % D[frame, 51])
            fid.write('rfoot %f %f\n' % tuple(D[frame, 52:54]))
            fid.write('rtoes %f\n' % D[frame, 54])
            fid.write('lfemur %f %f %f\n' % tuple(D[frame, 55:58]))
            fid.write('ltibia %f\n' % D[frame, 58])
            fid.write('lfoot %f %f\n' % tuple(D[frame, 59:61]))
            fid.write('ltoes %f\n' % D[frame, 61])


def build_CMU_sequence(sequence_path, save_path):
    """
    Generate training set for the planner method
    :param sequence_path: Path to the *.amc file
    :param save_path: Path where we save the *.txt file that will be used to train the planner 

    """

    seq = amc2txt(sequence_path, './data/walk.txt')

    np.savetxt(save_path, seq)


if __name__ == "__main__":
    build_CMU_sequence('./walk.amc', 'training_walk.txt')
    
