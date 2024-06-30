import numpy as np
from scipy.special import gamma
from scipy.integrate import quad
import matplotlib.pyplot as plt

class Fracode:

    '''
    Defines a system of three coupled Fractional Ordinary Differential Equations.\n 
    Supports calculating solutions numerically and parameter fitting to data.\n
    Suggested use: Testing SIR models to epidemiological data.
    '''

    def __init__(
            self, 
            system, 
            alpha: float, 
            i_vals = [0,0,0],
            pderivs = []
    ):

        self.system = system
        self.alpha = alpha
        self.i_vals = i_vals
        self.pderivs = pderivs

    def __str__(self):
        return f'Coupled system of {len(self.system(0, [0,0,0]))} ODEs with alpha = {self.alpha}'

    def emsolve(
            self, 
            domain: tuple, 
            params: list, 
            timesteps: int,
            alpha = -1.0
    ) -> list:
        
        '''
        Calculates a matrix of solution points of size dependent on input timestep.
        Uses Euler's Method with a numerical approximation of the Caputo derivative operator.
        '''
        if alpha == -1.0:
            alpha = self.alpha
        else:
            pass

        t = np.linspace(domain[0], domain[1], timesteps)
        u = np.zeros((len(t), 3))
        u[0] = np.array(self.i_vals)
        # define step size
        h = t[1] - t[0]
        
        # iterate for every time t, except for the initial value
        for i in range(1, len(t)):
            # Get the n value based on the already calculated u values
            n = i - 1

            # Summation portion
            z = np.zeros(n)
            for k in range(n):
                z[k] = (n+1-k)**(1-alpha) - (n-k)**(1-alpha)

            s_sum = np.sum([(u[j+1, 0] - u[j, 0])*(z[j]) for j in range(n)])
            i_sum = np.sum([(u[j+1, 1] - u[j, 1])*(z[j]) for j in range(n)])
            r_sum = np.sum([(u[j+1, 2] - u[j, 2])*(z[j]) for j in range(n)])

            # gamma portion from pdf
            gamma_part = (1-alpha)* gamma(1-alpha) * h**alpha
            caputo_ode = self.system(t[i], u[n], params)
            s_gamma_part = gamma_part * caputo_ode[0]
            i_gamma_part = gamma_part * caputo_ode[1]
            r_gamma_part = gamma_part * caputo_ode[2]

            # calculate u(t_{n+1})
            u[i, 0] = (u[i-1, 0] - s_sum + s_gamma_part)
            u[i, 1] = (u[i-1, 1] - i_sum + i_gamma_part)
            u[i, 2] = (u[i-1, 2] - r_sum + r_gamma_part)

        self.solution = u

        return self.solution
    

    def get_partials(
            self, 
            domain: tuple, 
            params: list,
            timesteps: int
    ):
        
        '''
        Returns the partial derivatives of I with respect to a, b, c, and alpha. 
        Used in grad_desc() and minimize().
        '''

        t = np.linspace(domain[0], domain[1], timesteps)
        x = self.emsolve(domain, params[:3], timesteps, params[3])
        h = t[1] - t[0]
        # Define partial derivatives
        # S
        dSda = np.zeros(len(t))
        dSdb = np.zeros(len(t))
        dSdc = np.zeros(len(t))
        dSdu = np.zeros(len(t))
        # I
        dIda = np.zeros(len(t))
        dIdb = np.zeros(len(t))
        dIdc = np.zeros(len(t))
        dIdu = np.zeros(len(t))
        # R
        dRda = np.zeros(len(t))
        dRdb = np.zeros(len(t))
        dRdc = np.zeros(len(t))
        dRdu = np.zeros(len(t))

        S = x[:, 0]
        I = x[:, 1]
        R = x[:, 2]

        # a = params[0]
        # b = params[1]
        # c = params[2]
        u = params[3]

        ### calculate partial derivatives numerically
        gamma_prime = quad(lambda t: (t**(-u)) * (np.exp(-t)) * (np.log(t)), 0, 20)[0]
        gamma_prime = round(gamma_prime, 10)
        
        for i in range(len(t)-1):
            n = i - 1

            z = np.zeros(n+1)
            w = np.zeros(n+1)
            for k in range(n):
                # storing common expressions to be indexed for efficiency
                z[k] = (n+1-k)**(1-u) - (n-k)**(1-u)
                w[k] = (((n-k)**(1-u))*(np.log(n-k))) - ((n+1-k)**(1-u)) * (np.log(n+1-k))
            
            universal_gamma_part = ((gamma(1-u))*(-h**(u) + (1-u)*(h**(u))*np.log(u)) - ((1-u)*(h**(u)) * gamma_prime))
            universal_gamma_part_2 = ((1-u) * gamma(1-u) * (h**(u)))

            # using partial_derivative() to calculate the variable part
            funcs = [S[i], I[i], R[i]]
            partials_list = [[dSdu[i], dSda[i], dSdb[i], dSdc[i]], [dIdu[i], dIda[i], dIdb[i], dIdc[i]], [dRdu[i], dRda[i], dRdb[i], dRdc[i]]]
            caputo_vector = self.system(t, funcs)
            partials_matrix = self.pderivs(t, funcs, partials_list)

            # partials of S
            dSdu_sum_part = dSdu[i] - np.sum([((dSdu[j+1] - dSdu[j])*z[j] + ((S[j+1] - S[j])*(w[j]))) for j in range(n)])
            dSdu_gamma_part = caputo_vector[0] * universal_gamma_part + universal_gamma_part_2 * partials_matrix[0][0]
            
            dSdu[i+1] = dSdu_sum_part + dSdu_gamma_part
            dSda[i+1] = dSda[i] - np.sum([(dSda[j+1] - dSda[j])*(z[j]) for j in range(n)]) + universal_gamma_part_2 * partials_matrix[0][1]
            dSdb[i+1] = dSdb[i] - np.sum([(dSdb[j+1] - dSdb[j])*(z[j]) for j in range(n)]) + universal_gamma_part_2 * partials_matrix[0][2]
            dSdc[i+1] = dSdc[i] - np.sum([(dSdc[j+1] - dSdc[j])*(z[j]) for j in range(n)]) + universal_gamma_part_2 * partials_matrix[0][3]
            
            # partials of I
            dIdu_sum_part = dIdu[i] - np.sum([((dIdu[j+1] - dIdu[j])*(z[j]) + ((I[j+1] - I[j])*(w[j]))) for j in range(n)])
            dIdu_gamma_part = caputo_vector[1] * universal_gamma_part + universal_gamma_part_2 * partials_matrix[1][0]
            
            dIdu[i+1] = dIdu_sum_part + dIdu_gamma_part
            dIda[i+1] = dIda[i] - np.sum([(dIda[j+1] - dIda[j])*(z[j]) for j in range(n)]) + universal_gamma_part_2 * partials_matrix[1][1] 
            dIdb[i+1] = dIdb[i] - np.sum([(dIdb[j+1] - dIdb[j])*(z[j]) for j in range(n)]) + universal_gamma_part_2 * partials_matrix[1][2]
            dIdc[i+1] = dIdc[i] - np.sum([(dIdc[j+1] - dIdc[j])*(z[j]) for j in range(n)]) + universal_gamma_part_2 * partials_matrix[1][3]
            
            # partials of R
            dRdu_sum_part = dRdu[i] - np.sum([((dRdu[j+1] - dRdu[j])*(z[j]) + ((R[j+1] - R[j])*(w[j]))) for j in range(n)])
            dRdu_gamma_part = caputo_vector[2] * universal_gamma_part + universal_gamma_part_2 * partials_matrix[2][0]
            
            dRdu[i+1] = dRdu_sum_part + dRdu_gamma_part
            dRda[i+1] = dRda[i] - np.sum([(dRda[j+1] - dRda[j])*(z[j]) for j in range(n)]) + universal_gamma_part_2 * partials_matrix[2][1]
            dRdb[i+1] = dRdb[i] - np.sum([(dRdb[j+1] - dRdb[j])*(z[j]) for j in range(n)]) + universal_gamma_part_2 * partials_matrix[2][2]
            dRdc[i+1] = dRdc[i] - np.sum([(dRdc[j+1] - dRdc[j])*(z[j]) for j in range(n)]) + universal_gamma_part_2 * partials_matrix[2][3]
        
        partials = np.array([dIda, dIdb, dIdc, dIdu])

        return partials
    

    def grad_desc_step(
            self, 
            data: list, 
            domain: tuple, 
            params: list, 
            learning_rate: float,
            timesteps: int,
            fixed_alpha = False
    ) -> list:
        
        '''
        Calculates the next step in a gradient descent algorithm using the loss function L = (x - x_hat)^2
        '''

        real_timesteps = (len(data)-1) * timesteps + 1
        partials = self.get_partials(domain, params, real_timesteps)
        x = self.solution
        # print(self.solution)

        # Unpack partials
        dIda = partials[0]
        dIdb = partials[1]
        dIdc = partials[2]
        dIdu = partials[3]

        # Define loss functions
        dlda_I = 0.0
        dldb_I = 0.0
        dldc_I = 0.0
        dldu_I = 0.0

        # calculate the losses
        n = len(data)
        for i in range(1, n+1):
            # Calculate gradient of the loss function with respect to a
            dlda_I += 2*((x[(i-1)*timesteps, 1]-data[i-1]))*dIda[((i-1)*timesteps)]
            # Calculate gradient of the loss function with respect to b
            dldb_I += 2*((x[(i-1)*timesteps, 1]-data[i-1]))*dIdb[((i-1)*timesteps)]
            # Calculate gradient of the loss function with respect to c
            dldc_I += 2*((x[(i-1)*timesteps, 1]-data[i-1]))*dIdc[((i-1)*timesteps)]
            # Calculate gradient of the loss function with respect to alpha
            dldu_I += 2*((x[(i-1)*timesteps, 1]-data[i-1]))*dIdu[((i-1)*timesteps)]

        ## Sum loss function
        loss_sum_I = np.sum([((x[(timesteps*z), 1]-data[z])**2) for z in range(n)])

        # Update the parameters in the opposite direction of the partial derivative of the loss function
        params[0] = params[0] - learning_rate*(1/n)*dlda_I
        params[1] = params[1] - learning_rate*(1/n)*dldb_I
        params[2] = params[2] - learning_rate*(1/n)*dldc_I

        if not fixed_alpha:
            # alpha direction (adjust magnifier porportionally to how quickly it should change)
            params[3] = params[3] - (1000)*learning_rate*(1/n)*dldu_I

        # calculate the loss average
        loss_I = loss_sum_I/n

        return [params[0], params[1], params[2], params[3], dldu_I, loss_I]
    

    def get_random_params(self, min, max):
        a = np.random.uniform(min, max)
        b = np.random.uniform(min, max)
        c = np.random.uniform(min, max)
        u = np.random.uniform(.2, 1.0)
        
        return [a, b, c, u]
    

    def minimize(
            self,
            data,
            timesteps: int,
            learning_rate: float,
            num_epochs: int,
            fixed_alpha = False,
            show_loss = False,
            show_params = False
    ):
        
        '''
        Finds parameters to fit the system to a dataset. Uses a combination of Monte Carlo and gradient descent algorithms.
        '''

        domain = (0, len(data))

        parameter_options = {}

        print('Searching for initial parameters...')
        for guess in range(50):

            params_guess = self.get_random_params(.01, .05)

            for epoch in range(3):
                if fixed_alpha:
                    descent = self.grad_desc_step(data, domain, params_guess, learning_rate, timesteps, True)
                else:
                    descent = self.grad_desc_step(data, domain, params_guess, learning_rate, timesteps)
                loss = descent[5]
                params_guess = descent[:4]
                #if loss < 100:
                    #learning_rate = .0000000005
                    #learning_rate /= 1.01
                #elif loss < .5:
                #    learning_rate /= 2

            if str(loss) != 'nan':
                parameter_options[loss] = params_guess

        min_loss = min(parameter_options.keys()) 
        params_guess = parameter_options[min_loss]

        print('Optimizing...')
        for epoch in range(num_epochs + 1):
            if fixed_alpha:
                descent = self.grad_desc_step(data, domain, params_guess, learning_rate, timesteps, True)
            else:
                descent = self.grad_desc_step(data, domain, params_guess, learning_rate, timesteps)
            params_guess = descent[:4]
            loss = descent[5]
            # print(f'{epoch} dldu_I is {params_guess[4]}')

            if show_loss:
                print(f'{epoch} I loss is {loss}')

            if show_params:
                # print(f'    parameter a: {params_guess[0]}')
                # print(f'    parameter b: {params_guess[1]}')
                # print(f'    parameter c: {params_guess[2]}')
                print(f'    parameter alpha: {params_guess[3]}')

            '''
            if loss < .5:
                learning_rate = .0000000002
                #learning_rate /= 1.1
            elif loss < 1:
                learning_rate = .0000000003
            '''

        return params_guess

    

def main():
    print('test')

if __name__ == '__main__':
    main()

