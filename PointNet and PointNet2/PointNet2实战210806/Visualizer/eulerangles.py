# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""" Module implementing Euler angle rotations and their conversions
实现欧拉角旋转及其转换的模块
See:
* http://en.wikipedia.org/wiki/Rotation_matrix
* http://en.wikipedia.org/wiki/Euler_angles
* http://mathworld.wolfram.com/EulerAngles.html
See also: *Representing Attitude with Euler Angles and Quaternions: A
Reference* (2006) by James Diebel. A cached PDF link last found here:
http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.110.5134
Euler's rotation theorem tells us that any rotation in 3D can be
described by 3 angles.  Let's call the 3 angles the *Euler angle vector*
and call the angles in the vector :math:`alpha`, :math:`beta` and
:math:`gamma`.  The vector is [ :math:`alpha`,
:math:`beta`. :math:`gamma` ] and, in this description, the order of the
parameters specifies the order in which the rotations occur (so the
rotation corresponding to :math:`alpha` is applied first).
In order to specify the meaning of an *Euler angle vector* we need to
specify the axes around which each of the rotations corresponding to
:math:`alpha`, :math:`beta` and :math:`gamma` will occur.
There are therefore three axes for the rotations :math:`alpha`,
:math:`beta` and :math:`gamma`; let's call them :math:`i` :math:`j`,
:math:`k`.

*用欧拉角和四元数表示姿态：James Diebel的参考*（2006）。上次在此处找到的缓存PDF链接：
http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.110.5134 欧拉旋转定理告诉我们，
3D中的任何旋转都可以用3个角度来描述。让我们把这3个角度称为*Euler角向量*，并把向量中的角度称为：
math:`alpha`，：math:`beta`和：math:`gamma`。向量为[：math:`alpha`，：math:`beta`:math:`gamma`]，
在本说明中，参数的顺序指定旋转发生的顺序（因此首先应用与：math:`alpha`相对应的旋转）。
为了指定*Euler角向量*的含义，我们需要指定每个旋转所围绕的轴，这些旋转对应于：
math:`alpha`、：math:`beta`和：math:`gamma`。因此，旋转有三个轴：math:`alpha`、
：math:`beta`和：math:`gamma`；让我们叫他们：math:`i`:math:`j`，：math:`k`。


Let us express the rotation :math:`alpha` around axis `i` as a 3 by 3
rotation matrix `A`.  Similarly :math:`beta` around `j` becomes 3 x 3
matrix `B` and :math:`gamma` around `k` becomes matrix `G`.  Then the
whole rotation expressed by the Euler angle vector [ :math:`alpha`,
:math:`beta`. :math:`gamma` ], `R` is given by::
   R = np.dot(G, np.dot(B, A))

让我们将旋转：数学：`alpha`绕轴`i`表示为3×3旋转矩阵`a`。类似地：math:`beta`about
`j`变成3x3矩阵`B`；math:`gamma`about`k`变成矩阵`G`。然后整个旋转由Euler角度向量[：
math:`alpha`，：math:`beta`]表示：数学：`gamma`]，`R`由：：R=np.dot（G，np.dot（B，A））给出

See http://mathworld.wolfram.com/EulerAngles.html
The order :math:`G B A` expresses the fact that the rotations are
performed in the order of the vector (:math:`alpha` around axis `i` =
`A` first).
顺序：math:`gba`表示按照向量的顺序执行旋转的事实（：math:`alpha`绕轴`i`=`A`首先）。

To convert a given Euler angle vector to a meaningful rotation, and a
rotation matrix, we need to define:
要将给定的Euler角向量转换为有意义的旋转和旋转矩阵，我们需要定义：
* the axes `i`, `j`, `k`
* whether a rotation matrix should be applied on the left of a vector to
  be transformed (vectors are column vectors) or on the right (vectors
  are row vectors).
  旋转矩阵是否应应用于待变换向量的左侧（向量为列向量）或右侧（向量为行向量）。

* whether the rotations move the axes as they are applied (intrinsic
  rotations) - compared the situation where the axes stay fixed and the
  vectors move within the axis frame (extrinsic)
  旋转是否在应用时移动轴（固有旋转）-比较轴保持固定和向量在轴帧内移动的情况（外在）

* the handedness of the coordinate system
坐标系的利手性

See: http://en.wikipedia.org/wiki/Rotation_matrix#Ambiguities
We are using the following conventions:
我们正在使用以下约定：
* axes `i`, `j`, `k` are the `z`, `y`, and `x` axes respectively.  Thus
  an Euler angle vector [ :math:`alpha`, :math:`beta`. :math:`gamma` ]
  in our convention implies a :math:`alpha` radian rotation around the
  `z` axis, followed by a :math:`beta` rotation around the `y` axis,
  followed by a :math:`gamma` rotation around the `x` axis.
  轴‘i’、‘j’、‘k’分别是‘z’、‘y’和‘x’轴。因此，欧拉角向量[：math:`alpha`、：math:`beta`:
  在我们的约定中，math:`gamma`]表示a:math:`alpha`弧度绕`z`轴旋转，后跟a:math:`beta`绕`y`轴旋转，
  后跟a:math:`gamma`绕`x`轴旋转。

* the rotation matrix applies on the left, to column vectors on the
  right, so if `R` is the rotation matrix, and `v` is a 3 x N matrix
  with N column vectors, the transformed vector set `vdash` is given by
  ``vdash = np.dot(R, v)``.
  旋转矩阵在左侧应用于右侧的列向量，因此，如果'R'是旋转矩阵，并且'v'是具有N个列向量的3 x N矩阵，
  则变换后的向量集'vdash'由``vdash=np.dot（R，v）``给出。

* extrinsic rotations - the axes are fixed, and do not move with the
  rotations.
  外部旋转-轴是固定的，不随旋转而移动。

* a right-handed coordinate system
The convention of rotation around ``z``, followed by rotation around
``y``, followed by rotation around ``x``, is known (confusingly) as
"xyz", pitch-roll-yaw, Cardan angles, or Tait-Bryan angles.
右手坐标系的惯例是围绕“z”旋转，然后围绕“y”旋转，然后围绕“x”旋转，这被称为（令人困惑的）
xyz、俯仰-滚转-偏航、卡丹角或泰特-布莱恩角。
"""

import math

import sys
if sys.version_info >= (3, 0):
    from functools import reduce

import numpy as np


_FLOAT_EPS_4 = np.finfo(float).eps * 4.0


def euler2mat(z=0, y=0, x=0):
    """ Return matrix for rotations around z, y and x axes
    Uses the z, then y, then x convention above
    围绕z、y和x轴旋转的返回矩阵使用上面的z、y、x约定
    Parameters
    ----------
    z : scalar 标量
       Rotation angle in radians around z-axis (performed first)
       以弧度表示的绕z轴旋转角度（先执行）
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)
    Returns
    -------
    M : array shape (3,3)
       Rotation matrix giving same rotation as for given angles
       旋转矩阵提供与给定角度相同的旋转
    Examples
    --------
    >>> zrot = 1.3 # radians
    >>> yrot = -0.1
    >>> xrot = 0.2
    >>> M = euler2mat(zrot, yrot, xrot)
    >>> M.shape == (3, 3)
    True
    The output rotation matrix is equal to the composition of the
    individual rotations
    输出旋转矩阵等于各个旋转的组合
    >>> M1 = euler2mat(zrot)
    >>> M2 = euler2mat(0, yrot)
    >>> M3 = euler2mat(0, 0, xrot)
    >>> composed_M = np.dot(M3, np.dot(M2, M1))
    >>> np.allclose(M, composed_M)
    True
    You can specify rotations by named arguments
    可以通过命名参数指定旋转
    >>> np.all(M3 == euler2mat(x=xrot))
    True
    When applying M to a vector, the vector should column vector to the
    right of M.  If the right hand side is a 2D array rather than a
    vector, then each column of the 2D array represents a vector.
    将M应用于向量时，该向量应为M右侧的列向量。如果右侧是二维数组而不是向量，则二维数组的每列表示一个向量。
    >>> vec = np.array([1, 0, 0]).reshape((3,1))
    >>> v2 = np.dot(M, vec)
    >>> vecs = np.array([[1, 0, 0],[0, 1, 0]]).T # giving 3x2 array
    >>> vecs2 = np.dot(M, vecs)
    Rotations are counter-clockwise.
    旋转方向为逆时针方向.
    >>> zred = np.dot(euler2mat(z=np.pi/2), np.eye(3))
    >>> np.allclose(zred, [[0, -1, 0],[1, 0, 0], [0, 0, 1]])
    True
    >>> yred = np.dot(euler2mat(y=np.pi/2), np.eye(3))
    >>> np.allclose(yred, [[0, 0, 1],[0, 1, 0], [-1, 0, 0]])
    True
    >>> xred = np.dot(euler2mat(x=np.pi/2), np.eye(3))
    >>> np.allclose(xred, [[1, 0, 0],[0, 0, -1], [0, 1, 0]])
    True
    Notes
    -----
    The direction of rotation is given by the right-hand rule (orient
    the thumb of the right hand along the axis around which the rotation
    occurs, with the end of the thumb at the positive end of the axis;
    curl your fingers; the direction your fingers curl is the direction
    of rotation).  Therefore, the rotations are counterclockwise if
    looking along the axis of rotation from positive to negative.
    旋转方向由右手尺给出（右手拇指沿着旋转发生的轴定向，拇指末端位于轴的正端；卷起你的手指；
    手指弯曲的方向是旋转的方向）。因此，如果沿着从正到负的旋转轴看，旋转是逆时针的。
    """
    Ms = []
    if z:
        cosz = math.cos(z)
        sinz = math.sin(z)
        Ms.append(np.array(
                [[cosz, -sinz, 0],
                 [sinz, cosz, 0],
                 [0, 0, 1]]))
    if y:
        cosy = math.cos(y)
        siny = math.sin(y)
        Ms.append(np.array(
                [[cosy, 0, siny],
                 [0, 1, 0],
                 [-siny, 0, cosy]]))
    if x:
        cosx = math.cos(x)
        sinx = math.sin(x)
        Ms.append(np.array(
                [[1, 0, 0],
                 [0, cosx, -sinx],
                 [0, sinx, cosx]]))
    if Ms:
        return reduce(np.dot, Ms[::-1])
    return np.eye(3)


def mat2euler(M, cy_thresh=None):
    """ Discover Euler angle vector from 3x3 matrix
    Uses the conventions above.
    从3x3矩阵中发现欧拉角矢量
    使用上述约定。
    Parameters
    ----------
    M : array-like, shape (3,3)
    cy_thresh : None or scalar, optional
       threshold below which to give up on straightforward arctan for
       estimating x rotation.  If None (default), estimate from
       precision of input.
       无或标量，可选
       低于该阈值时，放弃直接的arctan来估计x旋转。如果无（默认），则根据输入精度进行估计。
    Returns
    -------
    z : scalar 标量
    y : scalar
    x : scalar
       Rotations in radians around z, y, x axes, respectively
       分别围绕z、y、x轴以弧度旋转
    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::
    如果没有数值误差，则可使用z、y、x旋转矩阵的辛表达式导出例程，即：
      [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
      [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
      [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]
    with the obvious derivations for z, y, and x
    对于z，y和x有明显的导数(对z、y和x有明显的推导作用)?
       z = atan2(-r12, r11)
       y = asin(r13)
       x = atan2(-r23, r33)
    Problems arise when cos(y) is close to zero, because both of::
    当cos（y）接近零时会出现问题，因为：
       z = atan2(cos(y)*sin(z), cos(y)*cos(z))
       x = atan2(cos(y)*sin(x), cos(x)*cos(y))
    will be close to atan2(0, 0), and highly unstable.
    将接近atan2（0,0），且高度不稳定。
    The ``cy`` fix for numerical instability below is from: *Graphics
    Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
    0123361559.  Specifically it comes from EulerAngles.c by Ken
    Shoemake, and deals with the case where cos(y) is close to zero:
    下面对数值不稳定性的“cy”修复来自：*Graphics Gems IV*，Paul Heckbert（编辑），
    学术出版社，1994年，ISBN:0123361559。具体来说，它来自Ken Shoemake的EulerAngles.c，
    处理cos（y）接近于零的情况：

    See: http://www.graphicsgems.org/
    The code appears to be licensed (from the website) as "can be used
    without restrictions".
    该代码似乎被授权（从网站上）为“可以不受限制地使用”。
    """
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r33*r33 + r23*r23)
    if cy > cy_thresh: # cos(y) not close to zero, standard form
        z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
    else: # cos(y) (close to) zero, so x -> 0.0 (see above)
        # so r21 -> sin(z), r22 -> cos(z) and
        z = math.atan2(r21,  r22)
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = 0.0
    return z, y, x


def euler2quat(z=0, y=0, x=0):
    """ Return quaternion corresponding to these Euler angles
    Uses the z, then y, then x convention above
    返回与这些欧拉角对应的四元数
    使用上面的z、y、x约定
    Parameters
    ----------
    z : scalar 标量
       Rotation angle in radians around z-axis (performed first)
       以弧度表示的绕z轴旋转角度（先执行）
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)
    Returns
    -------
    quat : array shape (4,)
       Quaternion in w, x, y z (real, then vector) format
       四元数格式的向量
    Notes
    -----
    We can derive this formula in Sympy using:
    我们可以用Sympy推导这个公式
    1. Formula giving quaternion corresponding to rotation of theta radians
       about arbitrary axis:
       给出与θ弧度绕任意轴旋转对应的四元数的公式：：
       http://mathworld.wolfram.com/EulerParameters.html
    2. Generated formulae from 1.) for quaternions corresponding to
       theta radians rotations about ``x, y, z`` axes
       根据1.）生成的四元数公式，对应于θ弧度绕``x，y，z``轴旋转
    3. Apply quaternion multiplication formula -
       应用四元数乘法公式
       http://en.wikipedia.org/wiki/Quaternions#Hamilton_product - to
       formulae from 2.) to give formula for combined rotations.
       根据公式2.）给出组合旋转的公式。
    """
    z = z/2.0
    y = y/2.0
    x = x/2.0
    cz = math.cos(z)
    sz = math.sin(z)
    cy = math.cos(y)
    sy = math.sin(y)
    cx = math.cos(x)
    sx = math.sin(x)
    return np.array([
             cx*cy*cz - sx*sy*sz,
             cx*sy*sz + cy*cz*sx,
             cx*cz*sy - sx*cy*sz,
             cx*cy*sz + sx*cz*sy])


def quat2euler(q):
    """ Return Euler angles corresponding to quaternion `q`
    返回对应于四元数'q'的欧拉角`
    Parameters
    ----------
    q : 4 element sequence
       w, x, y, z of quaternion(四元数)
    Returns
    -------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)
    Notes
    -----
    It's possible to reduce the amount of calculation a little, by
    combining parts of the ``quat2mat`` and ``mat2euler`` functions, but
    the reduction in computation is small, and the code repetition is
    large.
    通过组合“quat2mat”和“mat2euler”函数的一部分，可以稍微减少计算量，但是计算量的减少很小，代码重复也很大。
    """
    # delayed import to avoid cyclic dependencies
    import nibabel.quaternions as nq
    return mat2euler(nq.quat2mat(q))


def euler2angle_axis(z=0, y=0, x=0):
    """ Return angle, axis corresponding to these Euler angles
    Uses the z, then y, then x convention above
    返回角，与这些欧拉角对应的轴
    使用上面的z、y、x约定
    Parameters
    ----------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)
    Returns
    -------
    theta : scalar
       angle of rotation
    vector : array shape (3,)
       axis around which rotation occurs
       旋转发生的轴
    Examples
    --------
    >>> theta, vec = euler2angle_axis(0, 1.5, 0)
    >>> print(theta)
    1.5
    >>> np.allclose(vec, [0, 1, 0])
    True
    """
    # delayed import to avoid cyclic dependencies
    import nibabel.quaternions as nq
    return nq.quat2angle_axis(euler2quat(z, y, x))


def angle_axis2euler(theta, vector, is_normalized=False):
    """ Convert angle, axis pair to Euler angles
    将角度、轴对转换为Euler角度
    Parameters
    ----------
    theta : scalar
       angle of rotation
    vector : 3 element sequence
       vector specifying axis for rotation.
       指定旋转轴的向量。
    is_normalized : bool, optional
       True if vector is already normalized (has norm of 1).  Default
       False
       如果向量已规格化（范数为1），则为True。默认False
    Returns
    -------
    z : scalar
    y : scalar
    x : scalar
       Rotations in radians around z, y, x axes, respectively
       分别围绕z、y、x轴以弧度旋转
    Examples
    --------
    >>> z, y, x = angle_axis2euler(0, [1, 0, 0])
    >>> np.allclose((z, y, x), 0)
    True
    Notes
    -----
    It's possible to reduce the amount of calculation a little, by
    combining parts of the ``angle_axis2mat`` and ``mat2euler``
    functions, but the reduction in computation is small, and the code
    repetition is large.
    通过组合“angle_axis2mat”和“mat2euler”函数的一部分，可以稍微减少计算量，但是计算量的减少很小，代码重复也很大。
    """
    # delayed import to avoid cyclic dependencies
    import nibabel.quaternions as nq
    M = nq.angle_axis2mat(theta, vector, is_normalized)
    return mat2euler(M)
