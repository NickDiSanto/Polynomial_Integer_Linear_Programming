import numpy as np


def compute(A, b, s):
    b_minus_s = np.subtract(b, s)
    dims = len(b_minus_s)

    x = np.random.randint(low=0, high=3, size=(dims,))
    TEMP = 200000

    best_x = x.copy()
    closest = distance(A, b_minus_s, best_x)

    history = [closest]
    x_closest = closest.copy()
    i = 1

    while (i <= TEMP):
        i += 1

        if (np.random.randint(i)) < 100:
            temp_x = np.random.randint(low=0, high=5, size=(dims,))
        else:
            temp_x = mutate(x.copy())

        temp_closest = distance(A, b_minus_s, temp_x)

        if (temp_closest < x_closest):
            x = temp_x.copy()
            x_closest = temp_closest.copy()

            if (temp_closest < closest):
                best_x = x.copy()
                closest = temp_closest.copy()
                history.append(temp_closest)

    return best_x, history


def mutate(x):
    i = np.random.randint(low=0, high=len(x))
    x[i] += 1 if (x[i] == 0 or np.random.randint(2) > 0) else -1

    return x


def distance(A, b_minus_s, x):
    return np.linalg.norm(np.subtract(b_minus_s, np.dot(A, x)))


def optimize(x, A, b_minus_s, c):
    dims = len(x)
    possible = np.asarray(x)

    for i in range(len(x)):
        x_up, x_down = x.copy(), x.copy()
        x_up[i] += 1
        x_down[i] -= 1

        possible = np.vstack([possible, x_up])  # mutate
        possible = np.vstack([possible, x_down])

    A_dot_possible = np.tensordot(possible, A, axes=[1, 1])
    b_diff = np.sum(np.abs(np.subtract(b_minus_s, A_dot_possible)), axis=1)

    def sort_help(elem):
        return elem[0]

    sorted_zip = sorted(list(zip(b_diff, possible)), key=sort_help)
    evaluate = [sorted_zip[0][1], sorted_zip[1][1]]

    def twin(_item):
        return np.max(np.abs(np.subtract(evaluate[1:], _item))) >= 2

    for item in sorted_zip[2:]:
        if not twin(item[1]):
            evaluate += [item[1]]

        if len(evaluate) > dims:
            break

    ## evaluate has the vertices touching the closest point of the surrounding nd cube
    ## number of vertices needed is 2**n (we have n+1 points in evaluate already)

    ## 2d we have 3 need 1 more point (opposite of closest)
    ## 3d we have 4 need 4 more points (opposite of every point)
    ## 4d we have 5 need 11 more points

    target = np.asarray([1, 1])

    for _ in range(dims - 1):
        for item in evaluate[1:]:
            for i in range(len(item)):
                x_up, x_down = item.copy(), item.copy()
                x_up[i] += 1
                x_down[i] -= 1

                _up = np.sort(np.einsum('ij->i', np.abs(np.subtract(evaluate, x_up))))
                _down = np.sort(np.einsum('ij->i', np.abs(np.subtract(evaluate, x_down))))

                if np.array_equal(_up[:2], target):
                    evaluate = np.vstack([evaluate, x_up])  # mutate

                elif np.array_equal(_down[:2], target):
                    evaluate = np.vstack([evaluate, x_down])

    print('all corners')
    print(evaluate)

    print('solution')
    return evaluate[np.argmax(np.dot(evaluate, c))]


## best integer representation is 1 1 1 2 -<> best solution is...

# A=np.asarray([[2,9,5,0],[12,11,1,0],[4,8,6,1],[5,2,4,0]])
# b=np.asarray([15,25,20,12])
# s=np.asarray([0,0,0,0])
# c=np.asarray([10, 0, -1,2])


## best integer representation is 2 0 2 -<> best solution is...

A = np.asarray([[2, 9, 5], [12, 11, 1], [4, 8, 6]])
b = np.asarray([15, 25, 20])
s = np.asarray([0, 0, 0])
c = np.asarray([10, 0, -1])
x, h = compute(A, b, s)

print('closest corner')
print(x)
print('\n')
print(optimize(x, A, b, c))
