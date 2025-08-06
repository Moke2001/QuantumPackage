from MokeQuantumSimulation.State.StatePreparer.StatePreparer import state_preparer


def overlap_tenpy(state_0,state_1):
    psi_0=state_preparer(state_0,'tenpy')
    psi_1=state_preparer(state_1,'tenpy')
    return psi_1.mps.overlap(psi_0.mps)