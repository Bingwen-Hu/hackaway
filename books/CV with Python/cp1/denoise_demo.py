import numpy as np


def denoise(img, U_init, tolerance=0.1, tau=0.125, tv_weight=100):
    """An implementation of the Rudin-Osher-Fatemi (ROF) denoising model
    Args:
        img: noisy image (grayscale)
        U_init: initial guess for U
        tolerance: tolerance for stop criterion
        tau: step length
        tv_weight: weight of the TV-regularizing term

    Returns:
        denoised and detextured image, texture residual
    """

    width, height = img.shape

    # init
    U = U_init
    Px = img  # x-component to the dual field
    Py = img  # y-component to the dual field
    error = 1

    while (error > tolerance):
        U_old = U

        # gradient of primal variable
        grad_U_x = np.roll(U, -1, axis=1) - U  # x-component of U's gradient
        grad_U_y = np.roll(U, -1, axis=0) - U  # y-component of U's gradient

        # update the dual variable
        Px_new = Px + (tau / tv_weight) * grad_U_x
        Py_new = Py + (tau / tv_weight) * grad_U_y
        norm_new = np.maximum(1, np.sqrt(Px_new ** 2 + Py_new ** 2))

        Px = Px_new / norm_new  # update of x-component
        Py = Py_new / norm_new  # update of y-component

        # update the primal variable
        Rx_Px = np.roll(Px, 1, axis=1)  # right x-translation of x-component
        Ry_Py = np.roll(Py, 1, axis=0)  # right y-translation of y-component

        div_P = (Px - Rx_Px) + (Py - Ry_Py)  # divergence of the dual field
        U = img + tv_weight * div_P  # update of the primal variable

        # update of error
        error = np.linalg.norm(U - U_old) / np.sqrt(height * width)

    return U, img - U  # denoised image and texture residual
