from sage.all import *

z = QQbar.zeta(3)

for x_00 in [0,1]:
    for x_01 in [0,1]:
        for x_10 in [0,1]:
            for x_11 in [0,1]:

                dft_coefs = [[0,0],[0,0]]
                for u in [0,1]:
                    for v in [0,1]:
                        dft_coefs[u][v] += x_00*z**(-1*(u*0+v*0)) + x_01*z**(-1*(u*0+v*1)) + x_10*z**(-1*(u*1+v*0)) + x_11*z**(-1*(u*1+v*1))

                print(f'{[x_00, x_01]}')
                # print(f'{[x_10, x_11]} : {dft_coefs[0][0], dft_coefs[1][0]+dft_coefs[0][1], dft_coefs[1][1]}\n')
                print(f'{[x_10, x_11]} : {[x_00+x_01+x_10+x_11, x_00*x_11+x_10*x_01]}\n')


