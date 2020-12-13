# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:18:09 2020

@author: alina
"""

import tensorflow as tf
import numpy as np


def physical_model(xos, contact_points, normals, actions, friction, mu,
                   contact):
    # softly binarize the contact
    contact = tf.where(tf.greater_equal(contact, 0.5),
                       tf.ones_like(contact), tf.zeros_like(contact))
    # contact = tf.nn.sigmoid((40*contact)-20)
    contact = tf.reshape(contact, [-1, 1])

    # upscale the friction parameter to its coorect value
    fr = friction * 100.

    with tf.variable_scope('prediction'):
        # first calculate the distance between the contact point and
        # the object
        r = contact_points - xos
        rx = tf.slice(r, [0, 0], [-1, 1])
        ry = tf.slice(r, [0, 1], [-1, 1])

        vp, keep_contact = \
            get_contact_mode(rx, ry, actions, fr, mu, normals, contact)

        dx, dy, rot = get_vel_model(vp, rx, ry, fr)

        tr = tf.concat([dx, dy], axis=-1) * contact
        rot = rot * contact

    return tr, rot, tf.reshape(keep_contact, [-1, 1])


def get_vel_model(vp, rx, ry, fr2):
    with tf.variable_scope('calculate_velocity'):
        ux = tf.slice(vp, [0, 0], [-1, 1])
        uy = tf.slice(vp, [0, 1], [-1, 1])

        rx2 = tf.square(rx)
        ry2 = tf.square(ry)

        div = fr2 + rx2 + ry2

        tx_tmp = tf.multiply((fr2 + rx2), ux) + \
            tf.multiply(rx, tf.multiply(ry, uy))
        tx = tf.divide(tx_tmp, div)

        ty_tmp = tf.multiply((fr2 + ry2), uy) + \
            tf.multiply(rx, tf.multiply(ry, ux))
        ty = tf.divide(ty_tmp, div)

        rot_tmp = tf.multiply(rx, ty) - tf.multiply(ry, tx)
        rot = tf.divide(rot_tmp, fr2)

    return tx, ty, rot


def get_contact_mode(rx, ry, action, fr2, mu, normal, contact):
    # for calculate the boundary forces of the friction cone
    ang = mu/180.*np.pi

    normal_norm = tf.linalg.norm(normal, axis=-1)
    ok_normal = tf.greater(normal_norm, 1e-6)

    # if we don't have a normal, we simulate one to prevent nans
    normal_t = tf.where(ok_normal, normal, normal + tf.ones_like(normal))
    normal_t = normal_t/tf.linalg.norm(normal_t, axis=-1, keepdims=True)
    nx = tf.slice(normal_t, [0, 0], [-1, 1])
    ny = tf.slice(normal_t, [0, 1], [-1, 1])

    # check if the normal points towards the object
    dir_center = - tf.concat([rx, ry], axis=-1)
    dir_center_norm = tf.linalg.norm(dir_center, axis=-1, keepdims=True)
    dir_center = tf.where(tf.greater(tf.squeeze(dir_center_norm), 0.),
                          dir_center/dir_center_norm, dir_center)

    prod = tf.matmul(tf.reshape(dir_center, [-1, 1, 2]),
                     tf.reshape(normal_t, [-1, 2, 1]))
    # prevent nans if prod is slightly higher than 1 due to numerics
    prod = tf.clip_by_value(prod, -0.999999999, 0.999999999)
    n_ang = tf.acos(tf.reshape(prod, [-1]))

    # if the angle is greater than 90 degree, the normal is incorrect
    ok_normal = tf.logical_and(ok_normal,
                               tf.less(tf.abs(n_ang), np.pi/2.+0.1))

    # same for the push
    push_norm = tf.linalg.norm(action, axis=-1)
    push = tf.greater(push_norm, 1e-6)
    action_t = tf.where(push, tf.identity(action),
                        action + tf.ones_like(action))
    action_t = action_t/tf.linalg.norm(action_t, axis=-1, keepdims=True)
    ux_normed = tf.slice(action_t, [0, 0], [-1, 1])
    uy_normed = tf.slice(action_t, [0, 1], [-1, 1])

    sin1 = tf.sin(ang)
    cos = tf.cos(ang)
    t11 = tf.concat([cos[:, :, None], -sin1[:, :, None]], axis=-1)
    t12 = tf.concat([sin1[:, :, None], cos[:, :, None]], axis=-1)
    rot_mat1 = tf.concat(axis=1, values=[t11, t12])

    sin2 = tf.sin(-ang)
    t21 = tf.concat([cos[:, :, None], -sin2[:, :, None]], axis=-1)
    t22 = tf.concat([sin2[:, :, None], cos[:, :, None]], axis=-1)
    rot_mat2 = tf.concat(axis=1, values=[t21, t22])

    # rotate the normal to get the boundary forces
    fb1 = tf.matmul(rot_mat1, tf.reshape(normal_t, [-1, 2, 1]))
    fb2 = tf.matmul(rot_mat2, tf.reshape(normal_t, [-1, 2, 1]))

    fbx1, fby1 = tf.unstack(tf.reshape(fb1, [-1, 2]), axis=-1)
    fbx2, fby2 = tf.unstack(tf.reshape(fb2, [-1, 2]), axis=-1)

    # torque
    m1 = tf.multiply(rx, fby1[:, None]) - tf.multiply(ry, fbx1[:, None])
    m2 = tf.multiply(rx, fby2[:, None]) - tf.multiply(ry, fbx2[:, None])

    # calculate the velocity at the contact point induced by the
    # boundary-forces
    vx_tmp1 = tf.multiply(fr2, fbx1[:, None])
    vy_tmp1 = tf.multiply(fr2, fby1[:, None])
    vx_tmp2 = tf.multiply(fr2, fbx2[:, None])
    vy_tmp2 = tf.multiply(fr2, fby2[:, None])

    vbx1 = vx_tmp1 - tf.multiply(m1, ry)
    vby1 = vy_tmp1 + tf.multiply(m1, rx)
    vbx2 = vx_tmp2 - tf.multiply(m2, ry)
    vby2 = vy_tmp2 + tf.multiply(m2, rx)

    n1 = tf.sqrt(tf.square(vbx1)+tf.square(vby1))
    n2 = tf.sqrt(tf.square(vbx2)+tf.square(vby2))

    # if we have the slipping case, we need to find the correct
    # boundary velocity and the scaling factor
    ang1 = tf.divide(vbx1 * ux_normed + vby1 * uy_normed, n1)
    ang2 = tf.divide(vbx2 * ux_normed + vby2 * uy_normed, n2)

    # if the angle between the push and one of the boundarie
    #  velocities is greater than the angle between the two
    # boundary velocities, the push is sliding
    ang3 = tf.divide(vbx2 * vbx1 + vby2 * vby1, n1 * n2)

    b1 = tf.concat([vbx1, vby1], axis=1)
    b2 = tf.concat([vbx2, vby2], axis=1)

    vb = tf.where(tf.squeeze(tf.greater_equal(ang1, ang2)), b1, b2)

    vbx = tf.slice(vb, [0, 0], [-1, 1])
    vby = tf.slice(vb, [0, 1], [-1, 1])

    kappa = tf.divide(nx * action[:, 0:1] + ny * action[:, 1:],
                      tf.multiply(nx, vbx) + tf.multiply(ny, vby))
    sticking = tf.logical_and(tf.less_equal(ang3, ang1),
                              tf.less_equal(ang3, ang2))
    vp = tf.multiply(kappa, vb)

    # check sticking or sliding
    vp_out = tf.where(tf.squeeze(sticking), action, vp)

    # if the normal or action were not properly defined, return the action
    # to not create any dependencies
    vp_out = tf.where(tf.logical_and(tf.squeeze(ok_normal),
                                     tf.squeeze(push)), vp_out, action)

    # check if the pusher moves away from the contact
    normed_a = action_t/tf.linalg.norm(action_t, axis=-1, keepdims=True)
    push_angle = tf.squeeze(tf.matmul(normed_a[:, None, :],
                                      normal_t[:, :, None]))

    lose_contact = tf.squeeze(tf.less(push_angle, -1e-2))
    # we can only break contact if there was contact in the first place
    lose_contact = tf.logical_and(lose_contact,
                                  tf.squeeze(tf.greater(contact, 0.)))
    # and if both normal and action were properly defined
    lose_contact = tf.logical_and(lose_contact, tf.squeeze(ok_normal))
    lose_contact = tf.logical_and(lose_contact, tf.squeeze(push))
    # in this case, the resulting push velocity is zero
    vp_out = tf.where(lose_contact, 0*vp_out, vp_out)

    return vp_out, tf.logical_not(lose_contact)


def physical_model_derivative(xos, contact_points, normals, actions, friction,
                              mu, contact):
    """
    Same as the above functions, but returns a jacobian for the ekf
    """
    bs = contact.get_shape()[0].value
    dim_x = 10
    # binarize the contact
    cont = tf.where(tf.greater_equal(contact, 0.5),
                    tf.ones_like(contact), tf.zeros_like(contact))

    with tf.variable_scope('prediction'):
        # first calculate the distance between the contact point and
        # the object
        r = contact_points - xos
        rx = tf.slice(r, [0, 0], [-1, 1])
        ry = tf.slice(r, [0, 1], [-1, 1])
        nx = tf.slice(normals, [0, 0], [-1, 1])
        ny = tf.slice(normals, [0, 1], [-1, 1])

        vp, dvpx, dvpy, keep_contact = \
            get_contact_mode_derivative(rx, ry, actions, friction, mu, nx,
                                        ny, cont)

        dvpxs = tf.unstack(dvpx, dim_x, axis=2)
        dvpys = tf.unstack(dvpy, dim_x, axis=2)
        dx, dy, rot, dddx, dddy, ddrot = \
            get_vel_model_derivative(vp, r, friction)

        # dvpx, dvpy, drx, dry, df
        dxs = tf.unstack(dddx, 5, axis=-1)
        dys = tf.unstack(dddy, 5, axis=-1)
        drs = tf.unstack(ddrot, 5, axis=-1)

        ddx = \
            tf.stack([cont*(-dxs[2] + dxs[0]*dvpxs[0] + dxs[1]*dvpys[0]),
                      cont*(-dxs[3] + dxs[0]*dvpxs[1] + dxs[1]*dvpys[1]),
                      tf.zeros([bs, 1], dtype=tf.float32),
                      cont*( dxs[4] + dxs[0]*dvpxs[3] + dxs[1]*dvpys[3]),
                      cont*(          dxs[0]*dvpxs[4] + dxs[1]*dvpys[4]),
                      cont*( dxs[2] + dxs[0]*dvpxs[5] + dxs[1]*dvpys[5]),
                      cont*( dxs[3] + dxs[0]*dvpxs[6] + dxs[1]*dvpys[6]),
                      cont*(          dxs[0]*dvpxs[7] + dxs[1]*dvpys[7]),
                      cont*(          dxs[0]*dvpxs[8] + dxs[1]*dvpys[8]),
                      #dcont*dx], axis=-1)
                      tf.zeros([bs, 1], dtype=tf.float32)], axis=-1)
        ddy = \
            tf.stack([cont*(-dys[2] + dys[0]*dvpxs[0] + dys[1]*dvpys[0]),
                      cont*(-dys[3] + dys[0]*dvpxs[1] + dys[1]*dvpys[1]),
                      tf.zeros([bs, 1], dtype=tf.float32),
                      cont*( dys[4] + dys[0]*dvpxs[3] + dys[1]*dvpys[3]),
                      cont*(          dys[0]*dvpxs[4] + dys[1]*dvpys[4]),
                      cont*( dys[2] + dys[0]*dvpxs[5] + dys[1]*dvpys[5]),
                      cont*( dys[3] + dys[0]*dvpxs[6] + dys[1]*dvpys[6]),
                      cont*(          dys[0]*dvpxs[7] + dys[1]*dvpys[7]),
                      cont*(          dys[0]*dvpxs[8] + dys[1]*dvpys[8]),
                      #dcont*dy], axis=-1)
                      tf.zeros([bs, 1], dtype=tf.float32)], axis=-1)
        dor = \
            tf.stack([cont*(-drs[2] + drs[0]*dvpxs[0] + drs[1]*dvpys[0]),
                      cont*(-drs[3] + drs[0]*dvpxs[1] + drs[1]*dvpys[1]),
                      tf.zeros([bs, 1], dtype=tf.float32),
                      cont*( drs[4] + drs[0]*dvpxs[3] + drs[1]*dvpys[3]),
                      cont*(          drs[0]*dvpxs[4] + drs[1]*dvpys[4]),
                      cont*( drs[2] + drs[0]*dvpxs[5] + drs[1]*dvpys[5]),
                      cont*( drs[3] + drs[0]*dvpxs[6] + drs[1]*dvpys[6]),
                      cont*(          drs[0]*dvpxs[7] + drs[1]*dvpys[7]),
                      cont*(          drs[0]*dvpxs[8] + drs[1]*dvpys[8]),
                      #dcont*rot], axis=-1)
                      tf.zeros([bs, 1], dtype=tf.float32)], axis=-1)

        tr = tf.concat([dx, dy], axis=-1) * cont
        rot = rot * cont

    return tr, rot, tf.reshape(keep_contact, [-1, 1]), ddx, ddy, dor


def get_vel_model_derivative(vp, contact_points, fr):
    with tf.variable_scope('calculate_velocity'):
        rx = tf.slice(contact_points, [0, 0], [-1, 1])
        rz = tf.slice(contact_points, [0, 1], [-1, 1])

        ux = tf.slice(vp, [0, 0], [-1, 1])
        uz = tf.slice(vp, [0, 1], [-1, 1])

        rx2 = tf.square(rx)
        rz2 = tf.square(rz)

        div = 100*fr + rx2 + rz2

        tx_tmp = tf.multiply((100*fr + rx2), ux) + \
            tf.multiply(rx, tf.multiply(rz, uz))
        tx = tf.divide(tx_tmp, div)

        tz_tmp = tf.multiply((100*fr + rz2), uz) + \
            tf.multiply(rx, tf.multiply(rz, ux))
        tz = tf.divide(tz_tmp, div)

        rot_tmp = tf.multiply(rx, tz) - tf.multiply(rz, tx)
        rot = tf.divide(rot_tmp, 100*fr)

        dxdf = (100*ux*div - 100*tx_tmp)/div**2
        dydf = (100*uz*div - 100*tz_tmp)/div**2

        dxdrx = ((2*rx*ux + rz*uz)*div - 2*rx*tx_tmp)/div**2
        dxdrz = (rx*uz*div - 2*rz*tx_tmp)/div**2
        dydrz = ((2*rz*uz + rx*ux)*div - 2*rz*tz_tmp)/div**2
        dydrx = (rz*ux*div - 2*rx*tz_tmp)/div**2

        dxdux = (100*fr + rx2)/div
        dxduz = (rx*rz)/div
        dyduz = (100*fr + rz2)/div
        dydux = (rx*rz)/div

        drdf = ((rx*dydf - rz*dxdf)*100*fr - 100*rot_tmp)/(100*fr)**2
        drdrx = (tz + rx*dydrx - rz*dxdrx)/(100*fr)
        drdrz = (rx*dydrz - tx - rz*dxdrz)/(100*fr)
        drdux = (rx*dydux - rz*dxdux)/(100*fr)
        drduz = (rx*dyduz - rz*dxduz)/(100*fr)

        # dvpx, dvpy, drx, dry, df
        dx = tf.stack([dxdux, dxduz, dxdrx, dxdrz, dxdf], axis=-1)
        dy = tf.stack([dydux, dyduz, dydrx, dydrz, dydf], axis=-1)
        drot = tf.stack([drdux, drduz, drdrx, drdrz, drdf], axis=-1)

    return tx, tz, rot, dx, dy, drot


def get_contact_mode_derivative(rx, rz, action, fr, mu, nnx, nny, contact):
    # dim_x = 10
    bs = rx.get_shape()[0]
    # fr2 = tf.square(fr)

    fri = 100 * fr
    # for calculate the boundary forces of the friction cone
    ang = mu/180.*np.pi
    # ang = tf.math.atan(mu)

    normal = tf.concat([nnx, nny], axis=-1)
    normal_norm = tf.linalg.norm(normal, axis=-1)
    ok_normal = tf.greater(normal_norm, 1e-6)
    # if we don't have a normal, we simulate one to prevent nans
    normal_t = tf.where(ok_normal, normal, normal + tf.ones_like(normal))
    normal_t = normal_t/tf.norm(normal_t, axis=-1, keepdims=True)
    nx = tf.slice(normal_t, [0, 0], [-1, 1])
    nz = tf.slice(normal_t, [0, 1], [-1, 1])

    # check if the normal points towards the object
    dir_center = - tf.concat([rx, rz], axis=-1)
    dir_center_norm = tf.linalg.norm(dir_center, axis=-1, keepdims=True)
    dir_center = tf.where(tf.greater(tf.squeeze(dir_center_norm), 0.),
                          dir_center/dir_center_norm, dir_center)
    prod = tf.matmul(tf.reshape(dir_center, [-1, 1, 2]),
                     tf.reshape(normal_t, [-1, 2, 1]))
    prod = tf.clip_by_value(prod, -0.999999999, 0.999999999)
    n_ang = tf.acos(tf.reshape(prod, [-1]))
    # # correct values over 180 deg.
    # n_ang = tf.where(tf.greater(tf.abs(n_ang), np.pi),
    #                  2*np.pi - tf.abs(n_ang), tf.abs(n_ang))
    # if the angle is greater than 90 degree, the normal is incorrect
    ok_normal = tf.logical_and(ok_normal,
                               tf.less(tf.abs(n_ang), np.pi/2. + 0.1))

    # same for the push
    push_norm = tf.linalg.norm(action, axis=-1)
    push = tf.greater(push_norm, 1e-6)
    action_t = tf.where(push, tf.identity(action),
                        action + tf.ones_like(action))
    uux = tf.slice(action_t, [0, 0], [-1, 1])
    uuz = tf.slice(action_t, [0, 1], [-1, 1])

    # new method using rotation matrix
    sin1 = tf.sin(ang)
    cos = tf.cos(ang)
    sin1 = tf.sin(ang)
    cos = tf.cos(ang)
    t11 = tf.concat([cos[:, :, None], -sin1[:, :, None]], axis=-1)
    t12 = tf.concat([sin1[:, :, None], cos[:, :, None]], axis=-1)
    rot_mat1 = tf.concat(axis=1, values=[t11, t12])

    sin2 = tf.sin(-ang)
    t21 = tf.concat([cos[:, :, None], -sin2[:, :, None]], axis=-1)
    t22 = tf.concat([sin2[:, :, None], cos[:, :, None]], axis=-1)
    rot_mat2 = tf.concat(axis=1, values=[t21, t22])

    # rotate the normal to get the boundary forces
    fb1 = tf.matmul(rot_mat1, tf.reshape(normal_t, [-1, 2, 1]))
    fb2 = tf.matmul(rot_mat2, tf.reshape(normal_t, [-1, 2, 1]))

    fbx1, fbz1 = tf.unstack(tf.reshape(fb1, [-1, 2]), axis=-1)
    fbx2, fbz2 = tf.unstack(tf.reshape(fb2, [-1, 2]), axis=-1)

    # torque
    m1 = tf.multiply(rx, fbz1[:, None]) - tf.multiply(rz, fbx1[:, None])
    m2 = tf.multiply(rx, fbz2[:, None]) - tf.multiply(rz, fbx2[:, None])

    # calculate the velocity at the contact point induced by the
    # boundary-forces
    vx_tmp1 = tf.multiply(fri, fbx1[:, None])
    vz_tmp1 = tf.multiply(fri, fbz1[:, None])
    vx_tmp2 = tf.multiply(fri, fbx2[:, None])
    vz_tmp2 = tf.multiply(fri, fbz2[:, None])

    omega1 = m1
    omega2 = m2

    vbx1 = vx_tmp1 - tf.multiply(omega1, rz)
    vbz1 = vz_tmp1 + tf.multiply(omega1, rx)
    vbx2 = vx_tmp2 - tf.multiply(omega2, rz)
    vbz2 = vz_tmp2 + tf.multiply(omega2, rx)

    # if we have the slipping case, we need to find the correct
    # boundary velocity and the scaling factor
    ang1 = tf.divide(tf.multiply(vbx1, uux)+tf.multiply(vbz1, uuz),
                     tf.multiply(tf.sqrt(tf.square(uux)+tf.square(uuz)),
                                 tf.sqrt(tf.square(vbx1)+tf.square(vbz1))))

    ang2 = tf.divide(tf.multiply(vbx2, uux)+tf.multiply(vbz2, uuz),
                     tf.multiply(tf.sqrt(tf.square(uux)+tf.square(uuz)),
                                 tf.sqrt(tf.square(vbx2)+tf.square(vbz2))))

    # if the angle between the push and one of the boundarie
    #  velocities is greater than the angle between the two
    # boundary velocities, the push is sliding
    ang3 = tf.divide(tf.multiply(vbx2, vbx1)+tf.multiply(vbz2, vbz1),
                     tf.multiply(tf.sqrt(tf.square(vbx1)+tf.square(vbz1)),
                                 tf.sqrt(tf.square(vbx2)+tf.square(vbz2))))

    vb = tf.where(tf.squeeze(tf.greater_equal(ang1, ang2)),
                  tf.concat([vbx1, vbz1], axis=1),
                  tf.concat([vbx2, vbz2], axis=1))

    vbx = tf.slice(vb, [0, 0], [-1, 1])
    vbz = tf.slice(vb, [0, 1], [-1, 1])

    kappa = tf.divide(tf.multiply(nx, uux) + tf.multiply(nz, uuz),
                      tf.multiply(nx, vbx) + tf.multiply(nz, vbz))

    sticking = tf.logical_and(tf.less_equal(ang3, ang1),
                              tf.less_equal(ang3, ang2))

    vp = tf.multiply(kappa, vb)

    # check sticking or sliding
    vp_out = tf.where(tf.squeeze(sticking), action, vp)

    # if the normal or action were not properly defined, return the action
    # to not create any dependencies
    vp_out = tf.where(tf.logical_and(tf.squeeze(ok_normal),
                                     tf.squeeze(push)), vp_out, action)

    # check if the pusher moves away from the contact
    normed_a = action_t/tf.linalg.norm(action_t, axis=-1, keepdims=True)
    push_angle = tf.squeeze(tf.matmul(normed_a[:, None, :],
                                      normal_t[:, :, None]))
    # happens at an angle of greater than 91 deg
    lose_contact = tf.squeeze(tf.less(push_angle, -1e-2))
    # we can only break contact if there was contact in the first place
    lose_contact = tf.logical_and(lose_contact,
                                  tf.squeeze(tf.greater(contact, 0.)))
    # and if both normal and action were properly defined
    lose_contact = tf.logical_and(lose_contact, tf.squeeze(ok_normal))
    lose_contact = tf.logical_and(lose_contact, tf.squeeze(push))

    # in this case, the resulting push velocity is zero
    vp_out = tf.where(lose_contact, 0*vp_out, vp_out)

    vpx = tf.slice(vp_out, [0, 0], [-1, 1])
    vpy = tf.slice(vp_out, [0, 1], [-1, 1])

    # gradients
    dvpx = tf.stack([tf.reshape(-tf.gradients(vpx, rx)[0], [bs, 1]),
                     tf.reshape(-tf.gradients(vpx, rz)[0], [bs, 1]),
                     tf.zeros([bs, 1]),
                     tf.reshape(tf.gradients(vpx, fr)[0], [bs, 1]),
                     tf.reshape(tf.gradients(vpx, mu)[0], [bs, 1]),
                     tf.reshape(tf.gradients(vpx, rx)[0], [bs, 1]),
                     tf.reshape(tf.gradients(vpx, rz)[0], [bs, 1]),
                     tf.reshape(tf.gradients(vpx, nnx)[0], [bs, 1]),
                     tf.reshape(tf.gradients(vpx, nny)[0], [bs, 1]),
                     tf.zeros([bs, 1])], axis=-1)

    dvpy = tf.stack([tf.reshape(-tf.gradients(vpy, rx)[0], [bs, 1]),
                     tf.reshape(-tf.gradients(vpy, rz)[0], [bs, 1]),
                     tf.zeros([bs, 1]),
                     tf.reshape(tf.gradients(vpy, fr)[0], [bs, 1]),
                     tf.reshape(tf.gradients(vpy, mu)[0], [bs, 1]),
                     tf.reshape(tf.gradients(vpy, rx)[0], [bs, 1]),
                     tf.reshape(tf.gradients(vpy, rz)[0], [bs, 1]),
                     tf.reshape(tf.gradients(vpy, nnx)[0], [bs, 1]),
                     tf.reshape(tf.gradients(vpy, nny)[0], [bs, 1]),
                     tf.zeros([bs, 1])], axis=-1)

    return vp_out, dvpx, dvpy, tf.logical_not(lose_contact)


###########################################################################
# projections between 2d and 3d
###########################################################################
def _to_2d(point, in_frame='world'):
    w2c = np.array([[0., 1., 0., 0.],
                    [0.66896468, -0., -0.74329412, -0.],
                    [-0.74329412, -0., -0.66896468, 0.67268115],
                    [0.0,  0.00,  0.0, 1.0]], dtype=np.float32)

    fx = 231.764480591
    fy = 231.76448822021484

    if in_frame != 'camera':
        point = tf.slice(point, [0, 0], [-1, 3])
        point = _to_cam_frame(point, w2c)

    xs = tf.slice(point, [0, 0], [-1, 1])
    ys = tf.slice(point, [0, 1], [-1, 1])
    zs = tf.slice(point, [0, 2], [-1, 1])

    # project
    out = [tf.divide(xs, zs) * fx, tf.divide(ys, zs)*fy]
    out = tf.concat(out, axis=1)
    return out


def _to_3d(point, image):
    w2c = np.array([[ 0., 1., 0., 0.],
                    [ 0.66896468, -0., -0.74329412, -0.],
                    [-0.74329412, -0., -0.66896468, 0.67268115],
                    [0.0,  0.00,  0.0, 1.0]], dtype=np.float32)

    c2w = np.linalg.inv(w2c)
    fx = 231.764480591
    fy = 231.76448822021484

    # fx = 289.7056007385254
    # fy = 289.70561027526855
    width = image.get_shape()[2].value
    height = image.get_shape()[1].value

    shape = point.get_shape()
    # get the z-value
    # grab 4 nearest corner points around the pixel coordinates
    coords_x = tf.slice(point, [0, 0], [-1, 1])
    coords_y = tf.slice(point, [0, 1], [-1, 1])

    x = coords_x + (width / 2.)
    y = coords_y + (height / 2.)

    x0s = tf.cast(tf.floor(x), 'int32')
    x1s = x0s + 1
    y0s = tf.cast(tf.floor(y), 'int32')
    y1s = y0s + 1

    # Limit the coordinates to be inside of the image
    x0s = tf.clip_by_value(x0s, 0, width-1)
    x1s = tf.clip_by_value(x1s, 0, width-1)
    y0s = tf.clip_by_value(y0s, 0, height-1)
    y1s = tf.clip_by_value(y1s, 0, height-1)

    zs = []
    for ind, b in enumerate(tf.unstack(image)):
        x_c = tf.unstack(x)[ind]
        y_c = tf.unstack(y)[ind]
        x0 = tf.unstack(x0s)[ind]
        x1 = tf.unstack(x1s)[ind]
        y0 = tf.unstack(y0s)[ind]
        y1 = tf.unstack(y1s)[ind]

        # transform the 4 corner points to indices in the
        # flattened source image
        base_y0 = y0*width
        base_y1 = y1*width
        idx_a = base_y1 + x1
        idx_b = base_y0 + x1
        idx_c = base_y1 + x0
        idx_d = base_y0 + x0

        # weighten each corner point according to its distance
        # to the actual target point
        x0_f = tf.cast(x0, 'float32')
        x1_f = tf.cast(x1, 'float32')
        y0_f = tf.cast(y0, 'float32')
        y1_f = tf.cast(y1, 'float32')

        wa = tf.multiply((x1_f - x_c), (y1_f - y_c))
        wb = tf.multiply((x1_f - x_c), (y_c - y0_f))
        wc = tf.multiply((x_c - x0_f), (y1_f - y_c))
        wd = tf.multiply((x_c - x0_f), (y_c - y0_f))

        # the interpolation weights should sum up to one (or zero)
        # so we normalize them
        norm = tf.add_n([wa, wb, wc, wd])
        binary_mask = tf.logical_and(tf.greater(norm, 0.), tf.less(norm, 1.))
        wa = tf.divide(wa, tf.where(binary_mask, norm,
                                    tf.ones_like(norm, dtype=tf.float32)))
        wb = tf.divide(wb, tf.where(binary_mask, norm,
                                    tf.ones_like(norm, dtype=tf.float32)))
        wc = tf.divide(wc, tf.where(binary_mask, norm,
                                    tf.ones_like(norm, dtype=tf.float32)))
        wd = tf.divide(wd, tf.where(binary_mask, norm,
                                    tf.ones_like(norm, dtype=tf.float32)))

        # use indices to lookup pixels in the flattened images
        flat = tf.reshape(b, [-1])
        flat = tf.cast(flat, 'float32')
        a = tf.gather(flat, idx_a)
        b = tf.gather(flat, idx_b)
        c = tf.gather(flat, idx_c)
        d = tf.gather(flat, idx_d)

        zs += [tf.math.abs(tf.reshape(tf.add_n([wa*a, wb*b, wc*c, wd*d]),
                                      [1]))]
    zs = tf.stop_gradient(tf.stack(zs))

    # unproject
    out = tf.concat([coords_x*zs/fx, coords_y*zs/fy,
                     zs, tf.constant(1., shape=[shape[0].value, 1])],
                    axis=1)

    # transform to world frame
    out = tf.matmul(c2w,
                    tf.expand_dims(out, -1))
    out = tf.reshape(out, [shape[0].value, 4])
    return tf.slice(out, [0, 0], [-1, 3])


def _to_3d_d(point, image, target):
    w2c = np.array([[0., 1., 0., 0.],
                    [0.66896468, -0., -0.74329412, -0.],
                    [-0.74329412, -0., -0.66896468, 0.67268115],
                    [0.0,  0.00,  0.0, 1.0]], dtype=np.float32)

    c2w = np.linalg.inv(w2c)
    fx = 231.764480591
    fy = 231.76448822021484

    width = image.get_shape()[2].value
    height = image.get_shape()[1].value

    target_cam = _to_cam_frame(target, w2c)

    shape = point.get_shape()
    # get the z-value
    # grab 4 nearest corner points around the pixel coordinates
    coords_x = tf.slice(point, [0, 0], [-1, 1])
    coords_y = tf.slice(point, [0, 1], [-1, 1])

    x = coords_x + (width / 2.)
    y = coords_y + (height / 2.)

    x0s = tf.cast(tf.floor(x), 'int32')
    x1s = x0s + 1
    y0s = tf.cast(tf.floor(y), 'int32')
    y1s = y0s + 1

    # Limit the coordinates to be inside of the image
    x0s = tf.clip_by_value(x0s, 0, width-1)
    x1s = tf.clip_by_value(x1s, 0, width-1)
    y0s = tf.clip_by_value(y0s, 0, height-1)
    y1s = tf.clip_by_value(y1s, 0, height-1)

    zs = []
    for ind, b in enumerate(tf.unstack(image)):
        x_c = tf.unstack(x)[ind]
        y_c = tf.unstack(y)[ind]
        x0 = tf.unstack(x0s)[ind]
        x1 = tf.unstack(x1s)[ind]
        y0 = tf.unstack(y0s)[ind]
        y1 = tf.unstack(y1s)[ind]

        # transform the 4 corner points to indices in the
        # flattened source image
        base_y0 = y0*width
        base_y1 = y1*width
        idx_a = base_y1 + x1
        idx_b = base_y0 + x1
        idx_c = base_y1 + x0
        idx_d = base_y0 + x0

        # weighten each corner point according to its distance
        # to the actual target point
        x0_f = tf.cast(x0, 'float32')
        x1_f = tf.cast(x1, 'float32')
        y0_f = tf.cast(y0, 'float32')
        y1_f = tf.cast(y1, 'float32')

        wa = tf.multiply((x1_f - x_c), (y1_f - y_c))
        wb = tf.multiply((x1_f - x_c), (y_c - y0_f))
        wc = tf.multiply((x_c - x0_f), (y1_f - y_c))
        wd = tf.multiply((x_c - x0_f), (y_c - y0_f))

        # the interpolation weights should sum up to one (or zero)
        # so we normalize them
        norm = tf.add_n([wa, wb, wc, wd])
        binary_mask = tf.logical_and(tf.greater(norm, 0.),
                                     tf.less(norm, 1.))
        wa = tf.divide(wa, tf.where(binary_mask, norm,
                                    tf.ones_like(norm, dtype=tf.float32)))
        wb = tf.divide(wb, tf.where(binary_mask, norm,
                                    tf.ones_like(norm, dtype=tf.float32)))
        wc = tf.divide(wc, tf.where(binary_mask, norm,
                                    tf.ones_like(norm, dtype=tf.float32)))
        wd = tf.divide(wd, tf.where(binary_mask, norm,
                                    tf.ones_like(norm, dtype=tf.float32)))

        # use indices to lookup pixels in the flattened images
        flat = tf.reshape(b, [-1])
        flat = tf.cast(flat, 'float32')
        a = tf.gather(flat, idx_a)
        b = tf.gather(flat, idx_b)
        c = tf.gather(flat, idx_c)
        d = tf.gather(flat, idx_d)

        zs += [tf.math.abs(tf.reshape(tf.add_n([wa*a, wb*b, wc*c, wd*d]),
                                      [1]))]
    zs = tf.stop_gradient(tf.stack(zs))

    diff = tf.abs(zs - target_cam[:, 2:])
    zs = tf.where(tf.greater(diff, 0.05), target_cam[:, 2:], zs)

    # unproject
    out = tf.concat([coords_x*zs/fx, coords_y*zs/fy,
                     zs, tf.constant(1., shape=[shape[0].value, 1])],
                    axis=1)

    # transform to world frame
    out = tf.matmul(c2w,
                    tf.expand_dims(out, -1))
    out = tf.reshape(out, [shape[0].value, 4])
    return tf.slice(out, [0, 0], [-1, 3])


def _to_world_frame(point, c2w):
    shape = point.get_shape()
    if shape[-1] < 4:
        point = tf.concat([point, tf.ones(shape=[shape[0].value, 1])],
                          axis=1)
    out = tf.matmul(c2w,
                    tf.expand_dims(point, -1))
    out = tf.reshape(out, [shape[0].value, 4])
    return tf.slice(out, [0, 0], [-1, 3])


def _to_cam_frame(point, w2c):
    shape = point.get_shape()
    if shape[-1] < 4:
        point = tf.concat([point, tf.ones(shape=[shape[0].value, 1])],
                          axis=1)
    out = tf.matmul(w2c,
                    tf.expand_dims(point, -1))
    out = tf.reshape(out, [shape[0].value, 4])
    return tf.slice(out, [0, 0], [-1, 3])
