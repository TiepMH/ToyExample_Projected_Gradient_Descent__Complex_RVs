''' Tiep M. Hoang '''
###############################################################################

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

###############################################################################
''' Tensorflow '''
def loss_tf(psi, beamforming):
    L = 0*tf.norm(psi)**2 + tf.norm(beamforming)**2
    return tf.real(L) 
    # tf.real(a + j0) = a to avoid the error when using tf.gradients

###############################################################################
''' Numpy '''
def loss(psi, beamforming):
    L = 0*np.linalg.norm(psi)**2  + np.linalg.norm(beamforming)**2 
    return np.real(L) 

def find__x_next(x_current, grad_at__x_current): 
    step_size = 0.1
    x = x_current - step_size * grad_at__x_current
    return x 

def Projected_Gradient_Descent_for_psi(psi,psi_region):
    psi_min = psi_region[0]
    psi_max = psi_region[1]
    psi = np.clip( psi, psi_min, psi_max ) 
    # For example, if x = np.array( [[ -1, 0.5, 3, 2, 0.7, 1 ]] ) ,
    # then np.clip returns x = np.array( [[ 0, 0.5, 1, 1, 0.7, 1 ]] )
    return psi 

def Projected_Gradient_Descent_for_beamforming(beam):
    squaredNorm_of_beamforming = np.linalg.norm(beam)**2
    if squaredNorm_of_beamforming > 1:
        beam = beam/np.linalg.norm(beam)
    else:
        beam = beam
    return beam 

###############################################################################
### CONSTRAINTs:
psi_region = [0,1] # psi = [psi_1, ..., psi_K] then 0 <= psi_k <= 1 

### Declare some tensorflow objects
psi_tf = tf.placeholder(tf.complex64, [1,2]) 
f_tf = tf.placeholder(tf.complex64, [1,1])
grad_tf = tf.gradients( loss_tf(psi_tf, f_tf), 
                        [psi_tf, f_tf] )

### Initialize the first value for the ROW VECTOR psi = [psi_1, psi_2] 
# and the first value for the beamforming f = f_re + 1j*f_im 
psi_current = np.array( [[0.3, 0.6]] )
f_current = np.array( [[-1.2 - 0.3j]] )
loss_current = loss( psi_current, np.array([0,0]) )

### Prepare for later use
psi_array = psi_current
f_array = f_current
loss_array = loss_current

''' Main Program '''
n_iterations = 20 
epsilon = 10**(-7)
count_break = 0 
for k in range(n_iterations):
    ### Use tf to evaluate the gradient at psi = psi_current
    grad_at__current_psi_and_beamforming = tf.Session().run(grad_tf, 
                                            feed_dict={ psi_tf: psi_current,
                                                        f_tf: f_current } )
    grad_at__current_psi = grad_at__current_psi_and_beamforming[0]
    grad_at__current_f = grad_at__current_psi_and_beamforming[1]

    ### Find the next point
    psi_next = find__x_next(psi_current, grad_at__current_psi)
    psi_next = Projected_Gradient_Descent_for_psi(psi_next, psi_region) 
    #
    f_next = find__x_next(f_current, grad_at__current_f)
    f_next = Projected_Gradient_Descent_for_beamforming(f_next)
    #
    loss_next = loss( psi_next, f_next )
    
    # ## Check if the algorithm converges
    # if np.abs( loss_current - loss_next ) < epsilon:
    #     count_break += 1
    #     break 
    # else:
    #     # Update the next point iteratively 
    #     psi_current = psi_next 
    #     beam_current = beam_next
    #     loss_current = loss_next 
    
    # Update the next point iteratively 
    psi_current = psi_next 
    f_current = f_next
    loss_current = loss_next 
    
    ### Store the results
    psi_array = np.append(psi_array, psi_current, axis=0) 
    f_array = np.append(f_array, f_current, axis=0) 
    loss_array = np.append(loss_array, loss_current)

### The obtained results
psi_1_array = psi_array[:,0] # the first column of psi_array, shape = [1, ?]
psi_2_array = psi_array[:,1] # the second column of psi_array, shape = [1, ?]
f_real_array = np.reshape( np.real(f_array) ,
                              [1, len(np.real(f_array))] ) # the real part of beamforming_array
f_imag_array = np.reshape( np.imag(f_array) ,
                              [1, len(np.real(f_array))] ) # the imag part of beamforming_array

###############################################################################
''' Illustrate the trajectory of beamforming vector f = f_re + j * f_im in 3D'''
fig1 = plt.figure(figsize=(8, 8))
ax1 = fig1.gca(projection='3d')

ax1.plot(f_real_array[0], f_imag_array[0], loss_array, label='Trajectory of $f\in\mathbb{C}$', 
        color='b', marker='*', markersize=5, linestyle='-', linewidth=2)

###############################################################################
''' Illustrate the loss function in 3D '''
def graph_of_loss(f_real, f_imag):
    L = 0 + (f_real**2 + f_imag**2) # 0 is related to psi, which isn't considered
    return np.real(L) 

beam_real_min, beam_real_max, beam_real_step = -0.8, 0.8, 0.01
beam_part_min, beam_part_max, beam_imag_step = -0.8, 0.8, 0.01
axis_1 = np.arange(beam_real_min, beam_real_max, beam_real_step) 
axis_2 = np.arange(beam_part_min, beam_part_max, beam_imag_step)

xx_1, xx_2 = np.meshgrid(axis_1, axis_2)
z = graph_of_loss(xx_1, xx_2) 
''' NOTE:
np.shape(z) = (?, ?). Do NOT write z = loss(axis_1, axis_2) because np.shape(z) = (1,?)
'''

# surf = ax1.plot_surface(xx_1, xx_2, z, alpha=1)
surf = ax1.plot_wireframe(xx_1, xx_2, z, alpha=.5, label='Surface of $L(f)$',
                          linestyle='-', rstride = 5, cstride = 5)
#cset = ax.contourf(xx_1, xx_2, z, zdir='z', offset=np.min(z), cmap=cm.viridis)


ax1.set_xlabel('Real part of $f$', fontsize=18)
ax1.set_ylabel('Imaginary part of $f$', fontsize=18)
ax1.set_zlabel('Loss $L(f)$', fontsize=18)
ax1.legend(loc='best', fontsize=12)
ax1.set_title('Find $f= f_{re} + j*f_{im}$ so that $L(f)$ is minimized', fontsize=15)

###############################################################################
### Check the constraint ||f||^2  <=  1. In this example, f is a scalar but not a vector,
# thus we have ||f||^2 = |f|^2 = f_re^2 + f_im^2 <= 1
print(r"The constraint is satisfied if ||f||^2 <= 1")
print(r"At the last iteration, we have ||f||^2 = ", np.linalg.norm(f_array[-1])**2 )

###############################################################################
### Save figures
plt.savefig('RESULT__Example_3.jpeg', dpi=100, bbox_inches = 'tight')