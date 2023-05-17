from controllers_utils import *


class Neuron:
    def __init__(self, timestep):
        self.dt = timestep
        self.vth = -40
        self.v_rest = -70  # Resting potential
        self.Rm = 600  # Membrane resistance
        self.taum = 60  # Membrane time constant
        self.tau_dec = 30  # Current decay time constant
        self.i_psp = 0.06  # Current inducing postsynaptic potential

        """
        Neuron defined by:
            - Voltage (float)
            - Spike (binary)
            - Input current (float)
            - Output current (float)             
        """
        self.v = self.v_rest
        self.spike = 0
        self.I_stim = 0
        self.I_out = 0

    def updateVoltage(self):
        """
            LIF neuron dynamics
        """
        V_inf = self.v_rest + self.I_stim * self.Rm
        self.v = V_inf + (self.v - V_inf) * np.exp(-self.dt / self.taum)
        if self.v > self.vth:
            self.spike = 1
            self.v = self.v_rest
        else:
            self.spike = 0

    def updateOutCurrent(self):
        """
            Update the neuron's output current
        """
        self.I_out -= self.dt*self.I_out/self.tau_dec
        if self.spike == 1:
            self.I_out += self.i_psp

    def updateInpCurrent(self, value):
        """
            Update the neuron's input current
        """
        self.I_stim = value


class MotorNeuronPool:
    """
    Neuron Pools defined in the same way as Neurons to use in the controller
    """
    def __init__(self, dt, size):
        self.dt = dt
        self.vth = -40
        self.v_rest = -70  # Resting potential
        self.Rm = 600  # Membrane resistance
        self.taum = 60  # Membrane time constant

        self.pool_wgts = np.random.uniform(low=0.4, high=0.6, size=size)

        self.v = self.v_rest*np.ones(size)
        self.spikes = np.zeros(size)
        self.I_stim = np.zeros(size)

    def updateVoltage(self):
        V_inf = self.v_rest + self.I_stim * self.Rm
        self.v = V_inf + (self.v - V_inf) * np.exp(-self.dt / self.taum)
        self.spikes[self.v >= self.vth] = 1
        self.spikes[self.v < self.vth] = 0
        self.v[self.v >= self.vth] = self.v_rest

    def get_pool_spikes(self):
        return np.sum(self.spikes)

    def updateInpCurrent(self, value):
        self.I_stim = value*self.pool_wgts

    def updateOutCurrent(self):
        pass


class AdaptiveSingleDOFController:
    def __init__(self, timestep, v_window, dq, q=0, Speed_Control=True, Facilitation=[1, 1], PSI=[1, 1], MN_pool_size=1):
        """
        Initialize a Single DOF Controller

        :param timestep: Simulation timestep to be used for neuron dynamics
        :param v_window: Time window for derivative estimation
        :param dq: Control resolution
        :param q: Initial value of the control variable (default is zero)
        """
        # Controller internal params
        self.theta = q
        self.theta_inc = dq
        self.dt = timestep

        self.speed_control_status = Speed_Control
        self.v_theta = 0
        self.theta_window = [q]
        self.v_theta_window = [self.v_theta]
        self.deriv_est_wind_len = v_window

        self.tau_fac = 500
        self.U_fac = 0.0025
        self.fac_status = Facilitation

        self.tau_PSI = 40
        self.U_PSI = 0.005
        self.PSI_status = PSI

        # Define neurons for the controller's SNN (4 mandatory + 2 + 1 optional)
        # [E, F, PPC_pos_speed, PPC_neg_speed, PPC_pos_error, PPC_neg_error, PSI]
        self.neurons = []
        for _ in range(2):
            self.neurons.append(MotorNeuronPool(self.dt, MN_pool_size))
        for _ in range(2):
            self.neurons.append(Neuron(self.dt))
        self.neuron_ids = {'E': 0,
                           'F': 1,
                           'PPC_pos_error': 2,
                           'PPC_neg_error': 3}

        if self.speed_control_status:
            for _ in range(2):
                self.neurons.append(Neuron(self.dt))
            self.neuron_ids['PPC_pos_speed'] = len(self.neuron_ids.keys())
            self.neuron_ids['PPC_neg_speed'] = len(self.neuron_ids.keys())

        if sum(self.PSI_status) > 0:
            self.neurons.append(Neuron(self.dt))
            self.neuron_ids['PSI'] = len(self.neuron_ids.keys())

        # Facilitation factors
        if self.fac_status[0] == 1:
            self.f3 = 0
            self.fac_max = 1
        else:
            self.f3 = 1
        if self.fac_status[1] == 1:
            self.f4 = 0
            self.fac_max = 1
        else:
            self.f4 = 1

        # Presynaptic Inhibition factors
        self.PSI3 = 0
        self.PSI4 = 0

    def update(self, theta_des):
        """
        Update the controller's state
        :param theta_des: Target value for the control variable at the specific timestep
        """
        for neuron in self.neurons:
            neuron.updateVoltage()  # Update Neuron Voltages according to LIF equations
            neuron.updateOutCurrent()  # Update Neurons' output currents

        # Update facilitation factors
        if self.fac_status[0] == 1:
            self.f3 -= self.dt * self.f3 / self.tau_fac
            if self.neurons[self.neuron_ids['PPC_pos_error']].spike == 1:
                self.f3 += self.U_fac
                self.f3 = min(self.f3, self.fac_max)
        if self.fac_status[1] == 1:
            self.f4 -= self.dt * self.f4 / self.tau_fac
            if self.neurons[self.neuron_ids['PPC_neg_error']].spike == 1:
                self.f4 += self.U_fac
                self.f4 = min(self.f4, self.fac_max)

        # Update Presynaptic Inhibition factors
        if self.PSI_status[0] == 1:
            self.PSI3 -= self.dt * self.PSI3 / self.tau_PSI
            if self.neurons[self.neuron_ids['PSI']].spike == 1:
                self.PSI3 += self.U_PSI
        if self.PSI_status[1] == 1:
            self.PSI4 -= self.dt * self.PSI4 / self.tau_PSI
            if self.neurons[self.neuron_ids['PSI']].spike == 1:
                self.PSI4 += self.U_PSI
        # Update angle
        self.theta += self.neurons[self.neuron_ids['E']].get_pool_spikes()*self.theta_inc
        self.theta -= self.neurons[self.neuron_ids['F']].get_pool_spikes()*self.theta_inc

        if self.speed_control_status:
            # Estimate derivative of control variable (speed)
            if len(self.theta_window) * self.dt == self.deriv_est_wind_len:
                self.theta_window.append(self.theta)
                self.theta_window.pop(0)
                self.v_theta = (self.theta_window[-1] - self.theta_window[0]) / self.deriv_est_wind_len
            else:
                self.theta_window.append(self.theta)
                self.v_theta = (self.theta_window[-1] - self.theta_window[0]) / (self.dt * len(self.theta_window))

            # Update Neurons' input currents
            a, b = v_theta2cur(self.v_theta)
            self.neurons[self.neuron_ids['PPC_pos_speed']].updateInpCurrent(a)
            self.neurons[self.neuron_ids['PPC_neg_speed']].updateInpCurrent(b)

        self.neurons[self.neuron_ids['PPC_pos_error']].updateInpCurrent(0.05 + 6*(ang2cur(self.theta) - ang2cur(theta_des) + ang2cur(-theta_des) - ang2cur(-self.theta)))
        self.neurons[self.neuron_ids['PPC_neg_error']].updateInpCurrent(0.05 + 6*(ang2cur(theta_des) - ang2cur(self.theta) - ang2cur(-theta_des) + ang2cur(-self.theta)))

        if sum(self.PSI_status) > 0:
            self.neurons[self.neuron_ids['PSI']].updateInpCurrent(5 * (self.neurons[self.neuron_ids['PPC_pos_error']].I_out + self.neurons[self.neuron_ids['PPC_neg_error']].I_out))
        max_psi_wgt = 2.75
        if self.PSI_status[0] == 1:
            adapt_wgt3 = PSI_wgt_clip(self.PSI3, max_psi_wgt)
        else:
            adapt_wgt3 = 1
        if self.PSI_status[1] == 1:
            adapt_wgt4 = PSI_wgt_clip(self.PSI4, max_psi_wgt)
        else:
            adapt_wgt4 = 1

        if self.speed_control_status:
            self.neurons[self.neuron_ids['E']].updateInpCurrent(3.5*self.neurons[self.neuron_ids['PPC_neg_speed']].I_out + adapt_wgt4*max_psi_wgt*self.f4*self.neurons[self.neuron_ids['PPC_neg_error']].I_out)
            self.neurons[self.neuron_ids['F']].updateInpCurrent(3.5*self.neurons[self.neuron_ids['PPC_pos_speed']].I_out + adapt_wgt3*max_psi_wgt*self.f3*self.neurons[self.neuron_ids['PPC_pos_error']].I_out)
        else:
            self.neurons[self.neuron_ids['E']].updateInpCurrent(adapt_wgt4*max_psi_wgt*self.f4*self.neurons[self.neuron_ids['PPC_neg_error']].I_out)
            self.neurons[self.neuron_ids['F']].updateInpCurrent(adapt_wgt3*max_psi_wgt*self.f3*self.neurons[self.neuron_ids['PPC_pos_error']].I_out)


class AgentController:
    def __init__(self, time_step, init_state, Speed_control=False, Facilitation=[0, 0], PSI=[0, 0], MN_pool_size=1, specs_list=None):
        """
        Define Agent Controller

        :param init_state: Vector with the initial state of each DOF of the agent (np.array)
        """
        self.dt = time_step  # integration timestep (ms)
        self.window_v = 20  # derivative estimation window
        self.theta_inc = 0.025  # angle change step

        if specs_list is None:
            self.DOF = len(init_state)
            self.DOF_ctrlrs = []
            for i in range(self.DOF):
                self.DOF_ctrlrs.append(AdaptiveSingleDOFController(timestep=self.dt,
                                                                v_window=self.window_v,
                                                                dq=self.theta_inc,
                                                                q=init_state[i],
                                                                Speed_Control=Speed_control,
                                                                Facilitation=Facilitation,
                                                                PSI=PSI,
                                                                MN_pool_size=MN_pool_size))
        else:
            self.DOF = len(init_state)
            self.DOF_ctrlrs = []
            for i in range(self.DOF):
                temp = np.where((specs_list[:, 0] == i))[0]  # get the specs of the trained controller
                if len(temp) > 0:
                    ind = temp[0]
                    joint_ctrlr_specs = specs_list[ind, 1:]
                else:
                    joint_ctrlr_specs = [0, 0, 0, 0, 0, 3]
                self.DOF_ctrlrs.append(AdaptiveSingleDOFController(timestep=self.dt,
                                                                v_window=self.window_v,
                                                                dq=self.theta_inc,
                                                                q=init_state[i],
                                                                Speed_Control=joint_ctrlr_specs[0],
                                                                Facilitation=joint_ctrlr_specs[1:3],
                                                                PSI=joint_ctrlr_specs[3:5],
                                                                MN_pool_size=joint_ctrlr_specs[5]))

    def update_state(self, q_des):
        """
        Update the state of the agent controller

        :param q_des: Vector with the desired state of the controller (np.array)
        """
        for k, DOF_ctrlr in enumerate(self.DOF_ctrlrs):
            DOF_ctrlr.update(q_des[k])

    def get_state(self):
        """
        Get the controller's state

        :return: Vector with each entry being the state of the controller of the corresponding DOF
        """
        return np.array([DOF_ctrlr.theta for DOF_ctrlr in self.DOF_ctrlrs])
        # joint_angles = np.array([DOF_ctrlr.theta for DOF_ctrlr in self.DOF_ctrlrs])
        # joint_vels = np.array([DOF_ctrlr.v_theta for DOF_ctrlr in self.DOF_ctrlrs])
        # return np.concatenate((joint_angles, joint_vels))








