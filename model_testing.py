import numpy as np
import matplotlib.pyplot as plt
from Fracode import Fracode

# define system of Caputo ODEs. u = [S, I, R]
def caputo_odes(t, u, params=[.03,.02,.01]):

    # more constants
    a = params[0]
    b = params[1]
    c = params[2]

    # assign each ODE to a vector element
    S = u[0]
    I = u[1]
    R = u[2]

    # system of ODEs
    dSdt = -a*S*I + c*R
    dIdt = a*S*I - b*I
    dRdt = b*I - c*R

    return [dSdt, dIdt, dRdt]



def partial_derivatives(t, funcs, pderivs, params=[.03, .02, .01, .5]):

    # constants
    a = params[0]
    b = params[1]
    c = params[2]
    u = params[3]

    # assign each function of t to a vector element
    S = funcs[0]
    I = funcs[1]
    R = funcs[2]

    # assign each pderivative of the funcs to a vector element
    dSdu = pderivs[0][0]
    dSda = pderivs[0][1]
    dSdb = pderivs[0][2]
    dSdc = pderivs[0][3]

    dIdu = pderivs[1][0]
    dIda = pderivs[1][1]
    dIdb = pderivs[1][2]
    dIdc = pderivs[1][3]

    dRdu = pderivs[2][0]
    dRda = pderivs[2][1]
    dRdb = pderivs[2][2]
    dRdc = pderivs[2][3]

    # Partial derivative functions

    next_dSdu = -a*((dSdu * I) + (S * dIdu)) + c * dRdu
    next_dSda = -(S*I + a*(S*dIda + I*dSda))+c*dRda
    next_dSdb = -a*(S*dIdb + I*dSdb) + c*dRdb
    next_dSdc = -a*(S*dIdc + I*dSdc) + R + c*dRdc

    next_dIdu = a*((dSdu * I) + (S * dIdu)) + b*dIdu
    next_dIda = S*I + a*(S*dIda+I*dSda) - b*dIda
    next_dIdb = a*(S*dIdb + I*dSdb) - (I + b*dIdb)
    next_dIdc = a*(S*dIdc + I*dSdc) - b*dIdc

    next_dRdu = b*dIdu - c*dRdu
    next_dRda = b*dIda - c*dRda
    next_dRdb = (I + b*dIdb) - c*dRdb
    next_dRdc = b*dIdc - (R + c*dRdc)


    return [[next_dSdu, next_dSda, next_dSdb, next_dSdc], [next_dIdu, next_dIda, next_dIdb, next_dIdc], [next_dRdu, next_dRda, next_dRdb, next_dRdc]]


def main():
    fractional_odes = Fracode(system=caputo_odes, alpha=0.5, i_vals=[99, 1, 0], pderivs=partial_derivatives)
    # solution = fractional_odes.emsolve((0, 1), [.03, .02, .01], 100, 0.5)
    # print(solution)

    #partials = fractional_odes.get_partials((0, 1), [.03, .02, .01, .5], 10)
    #print(partials)

    #data = [1, 24, 40]
    # data = [1, 2, 4, 8, 16, 32, 52, 63]
    data = [1, 2, 4, 7, 13, 17, 23, 31, 41, 56, 64, 69, 73, 77, 82, 85, 88, 91, 92, 93]
    domain = (0, len(data) - 1)
    timesteps = len(data) * 5 + 1

    #grad_desc = fractional_odes.grad_desc_step(data, (0, 2), [.03, .02, .01, .5], .00000000001, 5)
    #print(grad_desc)

    #best_fit = fractional_odes.minimize(data, 5, .00000000000005, 100, False, True, True)
    #print(best_fit)
    
    # print(fractional_odes)

    #model = Fracode(caputo_odes, best_fit[3], [99, 1, 0], partial_derivatives)
    model2 = Fracode(caputo_odes, 0.9243973973611928, [99, 1, 0], partial_derivatives)
    #solution = model.emsolve(domain, best_fit[:3], timesteps, best_fit[3])
    solution2 = model2.emsolve((0, 19), [0.018412947662674744, 0.049572007633482115, 0.04008640140794599], 101)

    '''
    S = solution[:,0]
    I = solution[:,1]
    R = solution[:,2]
    '''

    t_hat = np.linspace(0, len(data) - 1, timesteps)
    t_hat2 = np.linspace(0, 19, 101)
    t = np.linspace(0, len(data) - 1, len(data))

    # plot result
    '''
    plt.plot(t_hat, S, label="S")
    plt.plot(t_hat, I, label="I")
    plt.plot(t_hat, R, label="R")
    plt.plot(t, data, label="data")
    '''
    plt.plot(t_hat2, solution2[:,0], label="S")
    plt.plot(t_hat2, solution2[:,1], label="I")
    plt.plot(t_hat2, solution2[:,2], label="R")

    plt.xlabel('t')
    plt.ylabel('pop')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()