import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from cycler import cycler
from scipy.integrate import quad
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, Normalize

#Ahora si escribimos el sistema de ecuaciones completo.
def systemProca(r, cond, arg):
    p0x, p1x, p0y, p1y, p0z, p1z, u0x, u1x, u0y, u1y, u0z, u1z = cond
    LambT, = arg

    if r > 0:
        sumpi = p0x**2 + p0y**2 + p0z**2
        f0 = p1x
        f1 = LambT*sumpi*p0x - (2*p1x)/r - p0x*u0x
        f2 = p1y
        f3 = LambT*sumpi*p0y - (2*p1y)/r - p0y*u0y
        f4 = p1z
        f5 = LambT*sumpi*p0z - (2*p1z)/r - p0z*u0z
        f6 = u1x
        f7 = -sumpi-(2*u1x)/r
        f8 = u1y
        f9 = -sumpi - (2*u1y)/r
        f10 = u1z
        f11 = -sumpi - (2*u1z)/r
    else:
        sumpi = p0x**2 + p0y**2 + p0z**2
        f0 = p1x
        f1 = (LambT*sumpi*p0x - p0x*u0x)/3
        f2 = p1y
        f3 = (LambT*sumpi*p0y - p0y*u0y)/3
        f4 = p1z
        f5 = (LambT*sumpi*p0z - p0z*u0z)/3
        f6 = u1x
        f7 = -sumpi/3
        f8 = u1y
        f9 = -sumpi/3
        f10 = u1z
        f11 = -sumpi/3
        
    return [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11]
#Shooting para Proca
# shooting
def Shoot_Proca1(cond, Uintx, Uinty, Uintz, rmax, rmin=0, LambT=0, nodos=[0, 0, 0], 
                     met='RK45', Rtol=1e-09, Atol=1e-10, lim=1e-6, info=False, klim=500, outval=13, delta=0.4):
    nodos = np.array(nodos)
    p0x, p1x, p0y, p1y, p0z, p1z, u1x, u1y, u1z = cond

    p0Data = [p0x, p0y, p0z]

    Uminx, Umaxx = Uintx
    Uminy, Umaxy = Uinty
    Uminz, Umaxz = Uintz

    #print(f'Encontrando un perfil con {nodos}')

    #De nuevo definimos los eventos.
    def Sigx(r, U, arg): return U[0]
    def dSigx(r, U, arg): return U[1]
    def Sigy(r, U, arg): return U[2]
    def dSigy(r, U, arg): return U[3]
    def Sigz(r, U, arg): return U[4]
    def dSigz(r, U, arg): return U[5]
    Sigx.direction = 0; dSigx.direction = 0
    Sigy.direction = 0; dSigy.direction = 0
    Sigz.direction = 0; dSigz.direction = 0

    # ordenando de mayor a menor para la iteracion
    k = 0
    
    arg = [LambT]
    Uintrs = np.array([[Uminx, Umaxx], [Uminy, Umaxy], [Uminz, Umaxz]])
    sigModifold = None
    out = 0
    while True:
        u0 = np.array([shoot(*i) for i in Uintrs])
        V0 = [p0x, p1x, p0y, p1y, p0z, p1z, u0[0], u1x, u0[1], u1y, u0[2], u1z]
        
        sol = solve_ivp(systemProca, [rmin, rmax], V0, events=(Sigx, dSigx, Sigy, dSigy, Sigz, dSigz),
                         args=(arg,), method=met,  rtol=Rtol, atol=Atol)

        if info:
            plt.plot(sol.t, sol.y[0], c='red')
            plt.plot(sol.t, sol.y[2], c='blue')
            plt.plot(sol.t, sol.y[4], c='green')
            plt.xlim(0, 30)
            plt.ylim(-1.5, 1.5)
            plt.show()
  
        eventos = np.array([[sol.t_events[0], sol.t_events[1]],
                   [sol.t_events[2], sol.t_events[3]],
                   [sol.t_events[4], sol.t_events[5]]], dtype=object)
        sigModif = identify2(sol.t_events, nodos, p0Data, info=info)

        if info: 
            print(sigModif)

        iInterv, rTemp = freq_shoot(eventos[sigModif], nodos[sigModif], u0[sigModif], Uintrs[sigModif], rmax)
        
        if abs((iInterv[1]-iInterv[0])/2) <= lim:
            if info:
                print(out)
                print('Maxima precisión alcanzada: U0x = ', V0[6], ' U0y = ', V0[8], ' U0z = ', V0[10], 'radio', rTemp)
                return V0[6], V0[8], V0[10], rTemp, sol.t_events[0], sol.t_events[2], sol.t_events[4]
            if out==outval:
                print('Maxima precisión alcanzada: U0x = ', V0[6], ' U0y = ', V0[8], ' U0z = ', V0[10], 'radio', rTemp)
                return V0[6], V0[8], V0[10], rTemp, sol.t_events[0], sol.t_events[2], sol.t_events[4]
            else:
                #Uintrs = np.copy(UintrOrig) # reinicio los que  ya no son iguales
                if (iInterv[0] > delta):
                    Uintrs[sigModif] = [iInterv[0]-delta, iInterv[1]+delta]
                    out += 1
                else:
                    Uintrs[sigModif] = [0, iInterv[1]+delta] #quite el -delta
                    out += 1
        else:
            Uintrs[sigModif] = iInterv

        if info:
            print(Uintrs)

        if np.all(np.array([shoot(*i) for i in Uintrs])==u0): 
            print('Found: U0x = ', V0[6], ' U0y = ', V0[8], ' U0z = ', V0[10], 'radio', rTemp)
            return V0[6], V0[8], V0[10], rTemp, sol.t_events[0], sol.t_events[2], sol.t_events[4]
        
        if k==klim:
            print('loop limit reached')
            break
            
        k += 1

def shoot(imin, imax):
        return (imin+imax)/2   
def identify2(events, nodos, p0Data, info=False):
    """
    """
    dicNod = {'0': nodos[0], '2': nodos[1], '4': nodos[2]}

    # identificando que componentes no son ceros
    # cuando una componente se tomó como cero y se excluye del análisis
    indices = np.fromiter(map(int, dicNod.keys()), dtype=int)
    ind = list(map(bool, p0Data))
    posit = indices[ind]

    valR = [np.infty, np.infty, np.infty]
    for i in posit:
        #if info:
        #    print(events[i], events[i+1])
        
        valtemp = []
        nodo = dicNod[str(i)]
        numNod = events[i].size; numdSig = events[i+1].size
        if numNod == nodo and numdSig == nodo+1:
            valtemp.append(0)
        elif numNod == nodo:
            if numdSig < nodo+1:
                valtemp.append(events[i+1][nodo-1])
            elif numdSig > nodo+1:
                valtemp.append(events[i+1][nodo])
        elif numNod > nodo:
            valtemp.append(events[i][nodo])
        else:
            if numNod != 0:
                valtemp.append(events[i][-1])
            else:
                valtemp.append(0)

        if i==0:
            valR[0] = min(valtemp)
        elif i==2:
            valR[1] = min(valtemp)
        elif i==4:
            valR[2] = min(valtemp)
    
    # print(valR)
    sigModif = [False, False, False]
    test = np.min(valR)
    for i in range(3):
        if valR[i]==test:
            sigModif[i]=True
            break
    #print(sigModif)
    return sigModif
def identify(events, nodos, p0Data, info=False): #Esta función fue la que cambie más, personalmente no me gusta mucho usar diccionarios.
    """
    Analiza los eventos para ver cual componente requiere hacer shooting y regresa sigModif que es un lista con
     tres valores booleanos indicando si los nodos 0, 2, o 4 requieren atención. Notemos que estamos tratanto a p0Data como un booleano, esto no
     importa ya que valores no cero se evalúan como True y (0) se evalúa como False de manera automática.
    """
    # Claves y valores esperados de nodos
    nodos_clave = [0, 2, 4]
    nodos_indices = [nodos[0], nodos[1], nodos[2]]

    # Emparejamos los nodos con los datos inciales para ver cuales no son cero.
    nodos_activos = [idx for idx, flag in zip(nodos_clave, p0Data) if flag]
    # Inicializar valores con infinito, esto es porque filtramos buscando el mínimo.
    valR = [np.infty,np.infty,np.infty]

    for idx, nodo_pos in enumerate(nodos_activos):
        nodo = nodos_indices[idx]
        numNod = events[nodo_pos].size #numero de nodos
        numdSig = events[nodo_pos + 1].size #numero de nodos deseados en la derivada

        # Determinar valor mínimo  
        if numNod == nodo and numdSig == nodo + 1: #Si todo es correcto igualamos a cero
            val = 0
        elif numNod == nodo:
            val = (
                events[nodo_pos + 1][nodo - 1]
                if numdSig < nodo + 1
                else events[nodo_pos + 1][nodo]
            )
        elif numNod > nodo:
            val = events[nodo_pos][nodo]
        else:
            val = events[nodo_pos][-1] if numNod != 0 else 0

        # Actualizamos el valor mínimo
        valR[nodos_clave.index(nodo_pos)] = min(valR[nodos_clave.index(nodo_pos)], val)

    # Determinar cuál nodo requiere atención
    sigModif = [False, False, False]
    menor_valor = min(valR)
    for i in range(3):
        if valR[i] == menor_valor:
            sigModif[i] = True
            break
    #print(sigModif)
    return sigModif

#El shotting, cambie solo un poco para hacerla más fácil de leer (para mi gusto) y creo que una de las condiciones era redundante.
def freq_shoot(events, nodo, i0, iInterv, rTemp):
    imin, imax = iInterv[0]
    events = events[0]
    i0 = i0[0]

    # Si el tamaño cumple exactamente nodo y nodo+1, no se realizan ajustes
    if events[0].size == nodo and events[1].size == nodo + 1:
        return [imin, imax], rTemp

    # Condiciones para ajustar el intervalo y rTemp
    if events[0].size > nodo:
        imax = i0
        rTemp = events[0][-1]
    else:
        imin = i0
        rTemp = events[1][-1]
 
    return [imin, imax], rTemp

#MÉTODO REFINADO.
#Definimos primero el sistema de ecuaciones a resolver y la función para calcular las derivdas
#Matriz creada a partir de computar d/dz(\partial X/\partial c_i)ver 2208.13221v1.pdf. Lo que hacemos es derivar las distitnas f's del sistema original respecto a los parametros para almacenarlos en esta matríz
def MatrizDerivadas(r, cond, arg, info=False):
    LambT = arg
    p0x, p1x, p0y, p1y, p0z, p1z, u0x, u1x, u0y, u1y, u0z, u1z = cond
    sumpi = p0x**2 + p0y**2 + p0z**2
    # llenando matriz con las derivadas que conocemos.
    if r==0:
        Mat = np.zeros((12, 12))
    else:
        Mat = np.diag([0, -2/r]*6)  # lleno la diagonal
    #Para la diagonal secundaria
    diag1S = np.diag([1, 0]*6, k=1)[:-1,:-1]
    Mat += diag1S
    temp1 = np.array([-p0x, -p0y, -p0z])
    Mat[7::2, :5:2] = 2*temp1
    Mat[1, 6] = temp1[0]
    Mat[3, 8] = temp1[1]
    Mat[5, 10] = temp1[2]

    t1, t2, t3 = 2*LambT*p0y*p0x, 2*LambT*p0z*p0x, 2*LambT*p0z*p0y
    temp2 = [[LambT*(sumpi+2*p0x**2)-u0x, t1, t2], [t1, LambT*(sumpi+2*p0y**2)-u0y, t3],[t2, t3, LambT*(sumpi+2*p0z**2)-u0z]]
    Mat[1:6:2, :5:2] = temp2
    if r==0:
        #np.fill_diagonal(Mat, 0), recoordemos que en el caso r= 0 nos salia es 1/3
        Mat = Mat/3
    if info:
        print(Mat)
    return Mat
#Ya con la matriz definimos el nuevo sistema de ecuaciones a resolver de acuerdo con el paper
def system_refinado(r, condTot, arg):
    """
    condTot -> [p0x, p1x, p0y, p1y, p0z, p1z, u0x, u1x, u0y, u1y, u0z, u1z,\
           Duxp0x, Duxp1x, Duxp0y, Duxp1y, Duxp0z, Duxp1z, Duxu0x, Duxu1x, Duxu0y, Duxu1y, Duxu0z, Duxu1z,\
           Duyp0x, Duyp1x, Duyp0y, Duyp1y, Duyp0z, Duyp1z, Duyu0x, Duyu1x, Duyu0y, Duyu1y, Duyu0z, Duyu1z,\
           Duzp0x, Duzp1x, Duzp0y, Duzp1y, Duzp0z, Duzp1z, Duzu0x, Duzu1x, Duzu0y, Duzu1y, Duzu0z, Duzu1z]
    """

    cond = condTot[:12] #Las condiciones de frontera originales, son las primeras doce.
    cond_ux = condTot[12:24] #Las demás componentes
    cond_uy = condTot[24:36]
    cond_uz = condTot[36:]
    f_orig = systemProca(r, cond, arg=[arg])  #obtenemos las f's del sistema original
    
    #Ahora usando las f's originales obtenemos las derivadas
    Mat = MatrizDerivadas(r, cond, arg, info=False)
    #Hacemos el producto matricial para obtener las f's que nos faltan
    f12_23 = Mat@cond_ux  # Dux [f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23]
    f24_35 = Mat@cond_uy  # Duy [f24, f25, f26, f27, f28, f29, f30, f31, f32, f33, f34, f35]
    f36_47 = Mat@cond_uz  # Duz  [f36, f37, f38, f39, f40, f41, f42, f43, f44, f45, f46, f47]

    #Simplemente concatenamos los resultados.
    cond_new = np.concatenate((f_orig, f12_23, f24_35, f36_47))

    return cond_new
    
#Ya con las matrices podemos definir el sistema algebraico. El 22 del paper 2208.13221v1.pdf
def algebSyst(arg, remNul=True, info=False):
    """
    Resuelve el sistema algebraico 22 para ajustar las ck's
    Parámetros:
        - dXc: Matriz de derivadas parciales (coeficientes del sistema).
        - Xbc: Valores calculados en la iteracion actual
        - Xb: Valores esperados en la frontera
    """

    dXc, Xbc, Xb, ck = arg

    # Convertimos todo a numpy para evitar errores de compatibilidad
    dXc = np.array(dXc, dtype='float64') #Este es el lado izquierdo de la ecuación.
    Xbc = np.array(Xbc, dtype='float64')
    Xb = np.array(Xb, dtype='float64')
    ck = np.array(ck, dtype='float64')

    # Calcular MD (lado derecho del sistema)
    MD = np.dot(dXc, ck) - (Xbc - Xb)

    if remNul:
        # Identificar componentes no nulas
        nonzero_indices = MD != 0
        if not np.any(nonzero_indices):
            raise ValueError("MD es completamente nulo. No se puede resolver el sistema.")
        MI_reduced = dXc[:, nonzero_indices][nonzero_indices]
        MD_reduced = MD[nonzero_indices]

        
        # Resolver el sistema reducido
        ck1_reduced = np.linalg.solve(MI_reduced, MD_reduced)
    #Obtenemos le ck reducido 
        ck1 = np.zeros_like(ck)
        ck1[nonzero_indices] = ck1_reduced
    else:
        # Resolver el sistema completo
        ck1 = np.linalg.solve(dXc, MD)
    return ck1
#Ahora si tenemos todo lo necesario para hacer el fiting
def fitting(syst, V0, indck, indXc, BCind, inddXc, limit, argf=None, info=False,
            tol=1e-5, met='RK45', Rtol=1e-7, Atol=1e-8, npt=200, klim=25000):
    """
    syst: Sistema de ecuaciones diferenciales con la estructura f(r, yV, arg).
    V0 : Vector con las condiciones iniciales
    indck : Índices de los parámetros a ajustar dentro de V0.
    indXc : Índices de las variables con condiciones de contorno en el lado derecho.
    BCind : Condiciones de contorno.
    inddXc :  Índices de las derivadas asociadas a las variables con condiciones de contorno.
    klim :Máximo número de iteraciones """
 # llenamos el rango de integración
    rmin, rmax = limit
    rspan = np.linspace(rmin, rmax, npt)
    V0 = np.array(V0, dtype='float64')
    BCind = np.array(BCind, dtype='float64')

    # Iteración para ajuste
    for k in range(klim):
        # Resolver el sistema
        sol = solve_ivp(syst, [rmin, rmax], V0, t_eval=rspan, args=[argf], method=met, rtol=Rtol, atol=Atol)
        # Evaluar condiciones de contorno
        Xbc = sol.y[indXc, -1]
        error = np.abs(Xbc - BCind)
        if info:
            print(f"Iteración {k}, Error: {error}, Parámetros: {V0[indck]}")

        # Verificar si la solución cumple las condiciones de contorno
        if np.all(error < tol):
            print(f"Convergencia alcanzada en {k} iteraciones. con error: {error}")
            return V0

        # Calcular derivadas para ajuste
        temp = sol.y[inddXc, -1]
        dXc = temp.reshape((np.sum(indXc), np.sum(indck))).T #imp

        # Resolver sistema algebraico para ajustar las ck's
        arg = [dXc, Xbc, BCind, V0[indck]]
        ck = algebSyst(arg, info=info)
        # Actualizar valores iniciales
        V0[indck] = ck
    # Si no se alcanzó convergencia
    raise RuntimeError(f"No se alcanzó convergencia después de {klim} iteraciones. Error final: {error}")
#Calculo de masa y energía.
def energMul(r, sigtot, V0):
    
    sigF = interp1d(r, sigtot, kind='quadratic') #Primero obtenemos sigma por medio de una interpolación 
    def sigr(r):
        return r*sigF(r)
    def sigr2(r):
        return r**2*sigF(r)
    rmin = r[0]
    rfin = r[-1]
    #Integramos usando quad
    En = V0 - quad(sigr, rmin, rfin)[0]  # energía: (2c^2 m)/Lambda  -> Lambda=4pi m^3/Mp^2
    Mas = quad(sigr2, rmin, rfin)[0]  # masa: c*hb/(G*m*Lambda^(1/2))
    return En, Mas

#Ahora para la parte del funcional de energía dividimos la integral en dos, el primer termino y el segundo.
def Termino1(datos,gamma,rlim=None):
    r, sigma = datos
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma_scaled = r**gamma * sigma

    # Interpolaciones
    sigmaF = interp1d(r, sigma_scaled, kind='quadratic', fill_value="extrapolate")
    dsigma_scaled = np.gradient(sigma_scaled, r)
    dsigmaF = interp1d(r, dsigma_scaled, kind='quadratic', fill_value="extrapolate")

    # Límites de integración
    rmin, rfin = rlim if rlim else (r[0], r[-1])

    # Definir el integrando
    def integrand(r):
        dsigma_val = dsigmaF(r)
        sigma_val = sigmaF(r)
        return r**2 * (dsigma_val**2 + 2 * gamma * sigma_val**2 / r**2)

    #Integramos
    T1val = 2 * np.pi * quad(integrand, rmin, rfin)[0]
    return T1val
#El segundo termino
def Termino2(datos, gamma, rlim=None):
    # Validamos los tipos por si acaso y escalamos 
    r, sigma = datos
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma_scaled = r**gamma * sigma

    # Interpolamos 
    sigmaF = interp1d(r, sigma_scaled, kind='quadratic', fill_value="extrapolate")

    # Límites de integración
    rmin, rfin = rlim if rlim else (r[0], r[-1])

    # Definir el integrando
    def integrand(r):
        sigma_val = sigmaF(r)
        return r**2 * sigma_val**4

    #Inegramos
    T2val = np.pi * quad(integrand, rmin, rfin)[0]
    return T2val
#Ahora evaluamos el funcional de energía.
def E_Funcional(datos, arg, rlim=None):
    LambT, gamma = arg

    # Calculamos los dos terminos
    T1val = Termino1(datos, gamma, rlim=rlim)
    T2val = Termino2(datos, gamma, rlim=rlim)

    # Funcional de energía total
    Enf = -T1val - 2 * LambT * T2val
    return Enf, T1val, T2val
#Decidi escribir la función que computa la masa y energía de todos los datos acá, siento que es conveniente debido al shoting para los datos que haré despues
def MasaEnergia_Datos(datos, gamma, Nptos=1500, metodo='RK45', rtol=1e-9, atol=1e-10,sx=True,sy= True,sz= False,rec=False):
    resultados = []

    for i in datos:
        #print(i)
        # Parámetros iniciales
        px0, py0, pz0 = i[4]
        rmin, rmax = 0, i[3]
        LambT = i[5]

        rspan = np.linspace(rmin, rmax, Nptos)
        arg = [LambT]

        # Condiciones iniciales
        u0x, u0y, u0z = i[0], i[1], i[2]
        U0 = [px0, 0, py0, 0, pz0, 0, u0x, 0, u0y, 0, u0z, 0]
        # Resolver las ecuaciones diferenciales
        sol = solve_ivp(systemProca
            ,
            [rmin, rmax],
            U0,
            t_eval=rspan,
            args=(arg,),
            method=metodo,
            rtol=rtol,
            atol=atol
        )
        rad = sol.t
        px, ux = sol.y[0], sol.y[6][0]
        py, uy = sol.y[2], sol.y[8][0]
        pz, uz = sol.y[4], sol.y[10][0]
        
        # Cálculo de la masa total
        if (rec == True):
            recx = int((i[6][0])*Nptos/rmax)
            recy = int((i[6][1])*Nptos/rmax)
            recz = int((i[6][2])*Nptos/rmax)
            px_r = px if recx == 0 else px[:-recx]
            py_r = py if recy == 0 else py[:-recy]
            pz_r = pz if recz == 0 else pz[:-recz]
            rad_xr = rad if recx == 0 else rad[:-recx]
            rad_yr = rad if recz == 0 else rad[:-recy]
            rad_zr = rad if recz == 0 else rad[:-recz]
        else:
            px_r = px 
            py_r = py 
            pz_r = pz 
            rad_xr = rad 
            rad_yr = rad 
            rad_zr = rad 
        # Cálculo de la energía total
        arg2 = [LambT, gamma]
        datos_comps = [[rad_xr, px_r], [rad_yr, py_r], [rad_zr, pz_r]]
        EnT = sum(E_Funcional(datos_comp, arg2)[0] for datos_comp in datos_comps)
        #Masa
        Mx = Calc_Masa(rad_xr, px_r**2)
        My = Calc_Masa(rad_yr, py_r**2)
        Mz = Calc_Masa(rad_zr, pz_r**2)
        if (sz==False):
            Mz = 0
        if (sx==False):
            Mx = 0
        if (sy==False):
            My = 0
        MT = 4*np.pi*(Mx+My+Mz)
        # Guardar los resultados
        resultados.append([MT, EnT, px[0], py[0], pz[0]])

    return np.array(resultados)


def shoot_ref(Ini0, Uxint, Uyint, Uzint, rmax0,rext, LambT=0, nodos=[0, 0, 0],sx=True,sy=True,sz=False,semilla = False, ukk = [0, 0, 0, 0, [0, 0, 0],0,]):
    """
    Función que da la solución refinada, la masa y energía de forma directa directa.
        uk = rmax,[ux,uy,uz]
    Args:
        Ini0: Lista con las condiciones iniciales.
        Uxint, Uyint, Uzint: Semillas iniciales para las soluciones Ux, Uy, Uz.
        rmax0: Radio máximo inicial.
        LambT: Parámetro lambda de ajuste (por defecto 0).
        nodos: Lista de nodos iniciales para el shooting.
        sx, sy, sz: Indicadores para seleccionar cuál componente considerar (x, y o z).
    
    Returns:
        U0: Solución refinada inicial.
        rmax: Radio máximo tras refinamiento.
        masaT: Masa calculada.
        EnergiaT: Energía total calculada.
    """
    # Solución inicial no refinada
    if (semilla == False):
        U0x, U0y, U0z, rTmax, *_ = Shoot_Proca1(
            Ini0, Uxint, Uyint, Uzint, rmax0,
            LambT=LambT, nodos=nodos, lim=1e-8,
            info=False, klim=5000, outval=10, delta=0.2
        )
        p0x, p1x, p0y, p1y, p0z, p1z, u1x, u1y, u1z = Ini0
        U000 = [p0x,0,p0y,0,p0z, 0,U0x, 0,U0y,0,U0z,0]
        plotPerf(U000,rTmax,0)
    else: 
        U0x, U0y, U0z = ukk[0],ukk[1],ukk[2]
        p0x, p1x, p0y, p1y, p0z, p1z, u1x, u1y, u1z = ukk[4][0],0, ukk[4][1],0, ukk[4][2],0,0,0,0
        rTmax = ukk[3]
    # Definición del rango para el refinamiento
    rmin, rmax = 0, rTmax + rext
    limit = [rmin, rmax]
    
    #semillas no refinadas
    uk = [U0x, U0y, U0z]
    
    # Configuración inicial del vector V0 y sus derivadas
    V0X = [p0x, p1x, p0y, p1y, p0z, p1z, None, u1x, None, u1y, None, u1z]
    V0Duk = np.zeros(12 * 3)
    V0Duk[6] = 1  # Derivadas no triviales
    V0Duk[20] = 1
    V0Duk[-2] = 1
    V0 = np.concatenate((V0X, V0Duk))
    
    # Ajuste de las posiciones para las constantes ck
    indck = np.concatenate(([False, False, False, False, False, False, True, False, True, False, True, False], [False]*36))
    V0[indck] = uk
    
    # Índices de las variables para resolver y sus derivadas
    indXc = np.array([False] * 48)
    indXc[:5:2] = True  # Variables seleccionadas
    
    inddXc = np.array([False] * 48)
    inddXc[12::12] = True
    inddXc[14::12] = True
    inddXc[16::12] = True
    
    # Refinamiento usando fitting
    V0fit = fitting(
        system_refinado, V0, indck, indXc, [0, 0, 0], inddXc,
        limit, argf=LambT, tol=1e-10
    )
    
    # Cálculo de la masa y energía total
    metodo = 'RK45'
    Rtol, Atol = 1e-9, 1e-10
    Nptos = 500
    rspan = np.linspace(rmin, rmax, Nptos)
    U0 = V0fit[:12]
    
    sol = solve_ivp(
        systemProca, [rmin, rmax], U0, t_eval=rspan,
        args=([LambT],), method=metodo, rtol=Rtol, atol=Atol
    )
    
    rad = sol.t
    px, ux = sol.y[0], sol.y[6][0]
    py, uy = sol.y[2], sol.y[8][0]
    pz, uz = sol.y[4], sol.y[10][0]
    
    
    # Selección del componente para calcular energía y masa
    Ex, Mx = energMul(rad, px**2, ux)
    Ey, My = energMul(rad, py**2, uy)
    Ez, Mz = energMul(rad, pz**2, uz)
    if (sz==False):
        Mz,Ez = 0,0
    if (sx==False):
        Mx,Ex = 0,0
    if (sy==False):
        My,Ey = 0,0
    masaT = 4*np.pi*(Mx+My+Mz)
    EnergiaT = Ex + Ey + Ez
    return U0, rmax, masaT, EnergiaT
#################################################################################################################################################################
#Aqui es el shooting para concectar las soluciones.
"""def shoot_masa(n ,masa ,xmax, ymax, Uxint, Uyint, Uzint, rmax0, LambT, nodos,sx=True,sy=True,sz=False, tol = 0.05,kmax=20):
    #Lo ideal es empezar desde la izquierde e irnos moviendo hacia la derecha
    #Tomamos puntos en x que esten igualmente espaciados para solo hacer el shooting en y.
    p0z = 0
    ymin = 0
    ymax0 = ymax
    aprox = 0.5
    config = []
    j = 0
    while j < n:
        refinar = False
        #Primero obtenemos la solución no refinada y una masa aporximada
        k = 0
        ymin = 0
        ymax = ymax0
        p0x = np.random.random()*xmax
        while True:
            p0y = shoot(ymin,ymax)
            Ini0 = [p0x,0, p0y, 0, p0z, 0, 0, 0, 0]
            U0x, U0y, U0z, rTmax, *_ = Shoot_Proca1(Ini0, Uxint, Uyint, Uzint, rmax0,LambT=LambT, nodos=nodos, lim=1e-8,info=False, klim=1500, outval=10, 
                                                          delta=0.2)
            rmin, rmax = 0, rTmax
            Nptos = 500; rspan = np.linspace(rmin, rmax, Nptos); arg = [LambT]
            met = 'RK45'; Rtol = 1e-09; Atol = 1e-10
            U0 = [p0x, 0, p0y, 0, p0z, 0, U0x, 0, U0y, 0, U0z, 0]
            sol = solve_ivp(systemProca, [rmin, rmax], U0, t_eval=rspan,
                         args=(arg,), method=met, rtol=Rtol, atol=Atol)
            rad = sol.t
            px, ux = sol.y[0], sol.y[6][0]
            py, uy = sol.y[2], sol.y[8][0]
            pz, uz = sol.y[4], sol.y[10][0]
            #Calculamos la masa
            Mx = Calc_Masa(rad, px**2)
            My = Calc_Masa(rad, py**2)
            Mz = Calc_Masa(rad, pz**2)
            if (sz==False):
                Mz = 0
            if (sx==False):
                Mx = 0
            if (sy==False):
                My = 0
            masaT = 4*np.pi*(Mx+My+Mz)
            if (masaT > masa + aprox):
                ymax = p0y
            elif(masaT < masa - aprox):
                ymin = p0y
            else:
                refinar = True
                print(f"iteracion: {k} masa: {masaT}")
            if refinar:
                #Se cumplió la tolerancia aproximada, ahora podemos refinar para acercarnos más al valor de la masa
                # Definición del rango para el refinamiento
                rmin, rmax = 0, rTmax + 6
                limit = [rmin, rmax]
    
                # Variables iniciales y semillas no refinadas
                p1x, p1y, p1z, u1x, u1y, u1z = 0,0,0,0,0,0
                uk = [U0x, U0y, U0z]
                # Configuración inicial del vector V0 y sus derivadas
                V0X = [p0x, 0, p0y, 0, p0z, 0, None, u1y, None, u1y, None, u1z]
                V0Duk = np.zeros(12 * 3)
                V0Duk[6] = 1  # Derivadas no triviales
                V0Duk[20] = 1
                V0Duk[-2] = 1
                V0 = np.concatenate((V0X, V0Duk))
    
                # Ajuste de las posiciones para las constantes ck
                indck = np.concatenate(([False, False, False, False, False, False, True, False, True, False, True, False], [False]*36))
                V0[indck] = uk
    
                # Índices de las variables para resolver y sus derivadas
                indXc = np.array([False] * 48)
                indXc[:5:2] = True  # Variables seleccionadas
    
                inddXc = np.array([False] * 48)
                inddXc[12::12] = True
                inddXc[14::12] = True
                inddXc[16::12] = True
    
                # Refinamiento usando fitting
                V0fit = fitting(
                    system_refinado, V0, indck, indXc, [0, 0, 0], inddXc,
                    limit, argf=LambT, tol=1e-10
                )
    
                # Cálculo de la masa y energía total
                metodo = 'RK45'
                Rtol, Atol = 1e-9, 1e-10
                Nptos = 500
                rspan = np.linspace(rmin, rmax, Nptos)
                U0 = V0fit[:12]
    
                sol = solve_ivp(
                systemProca, [rmin, rmax], U0, t_eval=rspan,
                args=([LambT],), method=metodo, rtol=Rtol, atol=Atol
                )
    
                rad = sol.t
                px, ux = sol.y[0], sol.y[6][0]
                py, uy = sol.y[2], sol.y[8][0]
                pz, uz = sol.y[4], sol.y[10][0]
                 #Calculamos la masa
                Mx = Calc_Masa(rad, px**2)
                My = Calc_Masa(rad, py**2)
                Mz = Calc_Masa(rad, pz**2)
                if (sz==False):
                    Mz = 0
                if (sx==False):
                    Mx = 0
                if (sy==False):
                    My = 0
                masaT = 4*np.pi*(Mx+My+Mz)
                if (abs(masaT-masa)<tol):
                    config.append([U0[6], U0[8], U0[10], rmax, [U0[0], U0[2], U0[5]], nodos, LambT, masaT])
                    print("configuracion encontrada:")
                    print(config(j))
                    j += 1 
                    break
                if(k == kmax):
                    print("numero máximo de iteraciones")
                    break
                if (masaT > masa + tol):
                    ymax = p0y
                elif(masaT < masa - tol):
                    ymin = p0y
                print(masaT)
            k += 1
    return config"""
#El metodo anterior no parece funcionar porque las masas refindas y no refinadas son muy diferentes. Mejor se va a refinar todo
def shoot_masa(config,n ,masa ,xmin0=0,xmax0=1,ymin0=0, ymax0=1,z=0, Uxint=[0.1,2], Uyint=[0.1,2], Uzint=[0.1,2], rmax0=25,rmin0 =20, LambT=0, nodos=[0,1,0],sx=True,sy=True,sz=False, tol = 0.05,kmax=20,estacionario = False,recx=False,recy=False,recz=False,rext0=15,recorte = 0):
    #Lo ideal es empezar desde la izquierde e irnos moviendo hacia la derecha
    #Tomamos puntos en x que esten igualmente espaciados para solo hacer el shooting en y.
    p0z = z
    ymin = ymin0
    aprox = 0.5
    xarra = np.linspace(xmin0, xmax0, n)
    r00 = rmax0
    if estacionario:
        config, ymaxest = shoot_estacionario(masa, Uxint, Uyint, Uzint, rmax0, LambT, nodos,sx=False,sy=True,sz=False, tol = 0.05,kmax=20) 
        ymax0 = ymaxest + 0.5
    j = 0
    for p0x in xarra:
        #Primero obtenemos la solución no refinada y una masa aporximada
        k = 0
        ymin = ymin0
        ymax = ymax0
        while True:
            p0y = shoot(ymin,ymax)
            Ini0 = [p0x,0, p0y, 0, p0z, 0, 0, 0, 0]
            U0x, U0y, U0z, rTmax, *_ = Shoot_Proca1(Ini0, Uxint, Uyint, Uzint, r00,LambT=LambT, nodos=nodos, lim=1e-9,info=False, klim=1500, outval=10, 
                delta=0.2)
            rmin, rmax = 0, rTmax
            corte = rTmax
            if rTmax < rmin0:
                print("radio muy corto saltando valor de x")
                break
            # Definición del rango para el refinamiento
            rmin, rmax = 0, rTmax + rext0
            limit = [rmin, rmax]
            # Variables iniciales y semillas no refinadas
            p1x, p1y, p1z, u1x, u1y, u1z = 0,0,0,0,0,0
            uk = [U0x, U0y, U0z]
            # Configuración inicial del vector V0 y sus derivadas
            V0X = [p0x, 0, p0y, 0, p0z, 0, None, u1y, None, u1y, None, u1z]
            V0Duk = np.zeros(12 * 3)
            V0Duk[6] = 1  # Derivadas no triviales
            V0Duk[20] = 1
            V0Duk[-2] = 1
            V0 = np.concatenate((V0X, V0Duk))
    
            # Ajuste de las posiciones para las constantes ck
            indck = np.concatenate(([False, False, False, False, False, False, True, False, True, False, True, False], [False]*36))
            V0[indck] = uk
    
            # Índices de las variables para resolver y sus derivadas
            indXc = np.array([False] * 48)
            indXc[:5:2] = True  # Variables seleccionadas

            inddXc = np.array([False] * 48)
            inddXc[12::12] = True
            inddXc[14::12] = True
            inddXc[16::12] = True
    
            # Refinamiento usando fitting
            V0fit = fitting(
                system_refinado, V0, indck, indXc, [0,0,0], inddXc,
                limit, argf=LambT, tol=1e-6
                )
            # Cálculo de la masa y energía total
            metodo = 'RK45'
            Rtol, Atol = 1e-9, 1e-10
            Nptos = 2000
            rspan = np.linspace(rmin, rmax, Nptos)
            U0 = V0fit[:12]
    
            sol = solve_ivp(
            systemProca, [rmin, rmax], U0, t_eval=rspan,
                args=([LambT],), method=metodo, rtol=Rtol, atol=Atol
                )
            rad = sol.t
            px, ux = sol.y[0], sol.y[6][0]
            py, uy = sol.y[2], sol.y[8][0]
            pz, uz = sol.y[4], sol.y[10][0]
            #Calculamos la masa
            if recx:
                rec = int(rext0*Nptos/rmax)
                px = px[:-rec]
                rad_r = rad[:-rec]
            else:
                rad_r = rad
            if recy:
                recyy = int(recorte*Nptos/rmax)
                py = py[:-recyy]
                rad_ry = rad[:-recyy]
            else:
                rad_ry = rad
            if recz:
                reczz = int(recorte*Nptos/rmax)
                pz = pz[:-reczz]
                rad_rz = rad[:-reczz]
            else:
                rad_rz = rad
            Mx = Calc_Masa(rad_r, px**2)
            My = Calc_Masa(rad_ry, py**2)
            Mz = Calc_Masa(rad_rz, pz**2)
            if (sz==False):
                Mz = 0
            if (sx==False):
                Mx = 0
            if (sy==False):
                My = 0
            masaT = 4*np.pi*(Mx+My+Mz)
            if (abs(masaT-masa)<tol):
                #ux,uy,uz,rmax,[p0],lambT
                auxx = [U0[6], U0[8], U0[10], rmax, [U0[0], U0[2], U0[4]],LambT]
                config.append(auxx)
                print(f"configuracion encontrada con masa {masaT}:")
                print([U0[6], U0[8], U0[10], rmax, [U0[0], U0[2], U0[4]],LambT,[corte,0,0]])
                #Graficamos para asegurar que las soluciones sean validas
                U00 = [auxx[4][0],0,auxx[4][1],0,auxx[4][2], 0,auxx[0], 0,auxx[1],0,auxx[2],0]
                plotPerf(U00,auxx[3],0)
                plotPerf(U00,auxx[3],0,px=False,py=False,pz = True)
                plotPerf(U00,auxx[3],0,px=True,py=False,pz = False)
                plotPerf(U00,auxx[3],0,px=False,py=True,pz = False)
                j += 1 
                break
            if(k == kmax):
                print("numero máximo de iteraciones")
                break
            if (masaT > masa + tol):
                ymax = p0y
            elif(masaT < masa - tol):
                ymin = p0y
            auxx = [U0[6], U0[8], U0[10], rmax, [U0[0], U0[2], U0[4]],LambT]
            #Graficamos para asegurar que las soluciones sean validas
            U00 = [auxx[4][0],0,auxx[4][1],0,auxx[4][2], 0,auxx[0], 0,auxx[1],0,auxx[2],0]
            print(auxx)
            plotPerf(U00,auxx[3]-recorte,0,px=False,py=True,pz = False)
            plotPerf(U00,auxx[3]-recorte,0,px=False,py=False,pz = True)
            plotPerf(U00,auxx[3],0,px=True,py=False,pz = False)
            #plotPerf(U00,auxx[3],0,px=False,py=False,pz = True)
            print(masaT)
            k += 1
    #Intercambiamos elementos para que el estado excitado quede al final
    if estacionario:
        config[1], config[-1] = config[-1], config[1]
def shoot_randomz(n,xval=1,yval=1,zval = 0.08, yrange=0.1, Uxint=[0.1,2], Uyint=[0.1,2], Uzint=[0.1,2], rmax0=25, LambT=0, nodos=[0,1,0],sx=True,sy=True,sz=False, tol = 0.05,estacionario = False,rext0 = 6,recx=True,recy= True,recz=False,recorte = 5,rmin=0):
    #Lo ideal es empezar desde la izquierde e irnos moviendo hacia la derecha
    #Tomamos puntos en x que esten igualmente espaciados para solo hacer el shooting en y.
    p0z = zval
    aprox = 0.5
    r00 = rmax0
    p0x = xval
    j = 0
    while j < n:
        #Primero obtenemos la solución no refinada y una masa aporximada
        p0y = yval + np.random.random()*yrange
        Ini0 = [p0x,0, p0y, 0, p0z, 0, 0, 0, 0]
        U0x, U0y, U0z, rTmax, *_ = Shoot_Proca1(Ini0, Uxint, Uyint, Uzint, r00,LambT=LambT, nodos=nodos, lim=1e-8,info=False, klim=1500, outval=10, 
                                                            delta=0.2)
        rmin, rmax = 0, rTmax
        Nptos = 2000; rspan = np.linspace(rmin, rmax, Nptos); arg = [LambT]
        met = 'RK45'; Rtol = 1e-09; Atol = 1e-10
        U0 = [p0x, 0, p0y, 0, p0z, 0, U0x, 0, U0y, 0, U0z, 0]
        sol = solve_ivp(systemProca, [rmin, rmax], U0, t_eval=rspan,
                         args=(arg,), method=met, rtol=Rtol, atol=Atol)
        rad = sol.t
        px, ux = sol.y[0], sol.y[6][0]
        py, uy = sol.y[2], sol.y[8][0]
        pz, uz = sol.y[4], sol.y[10][0]
        print("no refinado")
        U000 = [p0x,0,p0y,0,p0z, 0,U0x, 0,U0y,0,U0z,0]
        if rTmax < rmin:
            print("radio muy corto")
            continue
        plotPerf(U000,rmax,0)
        # Definición del rango para el refinamiento
        rmin, rmax = 0, rTmax + rext0
        limit = [rmin, rmax]
         # Variables iniciales y semillas no refinadas
        p1x, p1y, p1z, u1x, u1y, u1z = 0,0,0,0,0,0
        uk = [U0x, U0y, U0z]
        # Configuración inicial del vector V0 y sus derivadas
        V0X = [p0x, 0, p0y, 0, p0z, 0, None, u1y, None, u1y, None, u1z]
        V0Duk = np.zeros(12 * 3)
        V0Duk[6] = 1  # Derivadas no triviales
        V0Duk[20] = 1
        V0Duk[-2] = 1
        V0 = np.concatenate((V0X, V0Duk))
    
        # Ajuste de las posiciones para las constantes ck
        indck = np.concatenate(([False, False, False, False, False, False, True, False, True, False, True, False], [False]*36))
        V0[indck] = uk
    
            # Índices de las variables para resolver y sus derivadas
        indXc = np.array([False] * 48)
        indXc[:5:2] = True  # Variables seleccionadas

        inddXc = np.array([False] * 48)
        inddXc[12::12] = True
        inddXc[14::12] = True
        inddXc[16::12] = True

        # Refinamiento usando fitting
        V0fit = fitting(
            system_refinado, V0, indck, indXc, [0,0,0], inddXc,
            limit, argf=LambT, tol=1e-7
            )
        # Cálculo de la masa y energía total
        metodo = 'RK45'
        Rtol, Atol = 1e-9, 1e-10
        Nptos = 2000
        rspan = np.linspace(rmin, rmax, Nptos)
        U0 = V0fit[:12]
    
        sol = solve_ivp(
        systemProca, [rmin, rmax], U0, t_eval=rspan,
                args=([LambT],), method=metodo, rtol=Rtol, atol=Atol
                )
        rad = sol.t
        px, ux = sol.y[0], sol.y[6][0]
        py, uy = sol.y[2], sol.y[8][0]
        pz, uz = sol.y[4], sol.y[10][0]
        #Calculamos la masa
        #Esto es para recortar px porque hay algunas veces que py se ve bien pero px se empieza a levantar pero su contribucion claramente deberia ser 0
        if recx:
            rec = int(rext0*Nptos/rmax)
            px_r = px[:-rec]
            rad_r = rad[:-rec]
        else:
            px_r = px
            rad_r = rad
        if recy:
            rec = int(recorte*Nptos/rmax)
            py_r = py[:-rec]
            rad_y = rad[:-rec]
        else:
            py_r = py
            rad_y = rad
        if recz:
            rec = int(recorte*Nptos/rmax)
            pz_r = pz[:-rec]
            rad_z = rad[:-rec]
        else:
            pz_r = pz
            rad_z = rad
        Mx = Calc_Masa(rad_r, px_r**2)
        My = Calc_Masa(rad_y, py_r**2)
        Mz = Calc_Masa(rad_z, pz_r**2)
        if (sz==False):
            Mz = 0
        if (sx==False):
            Mx = 0
        if (sy==False):
             My = 0
        masaT = 4*np.pi*(Mx+My+Mz)
        auxx = [U0[6], U0[8], U0[10], rmax, [U0[0], U0[2], U0[4]],LambT]
            #Graficamos para asegurar que las soluciones sean validas
        U00 = [auxx[4][0],0,auxx[4][1],0,auxx[4][2], 0,auxx[0], 0,auxx[1],0,auxx[2],0]
        print(auxx)
        plotPerf(U00,auxx[3],0,px=True,pz=True,py=True)
        plotPerf(U00,auxx[3]-recorte,0,px=False,pz=True,py=True)
        print(masaT)
        j +=1
def shoot_random2(n,xval=1,yval=1,zval = 0.08, yrange=0.1, Uxint=[0.1,2], Uyint=[0.1,2], Uzint=[0.1,2], rmax0=25, LambT=0, nodos=[0,1,0],sx=True,sy=True,sz=False, tol = 0.05,estacionario = False,rext0 = 6,div = 1):
    #Lo ideal es empezar desde la izquierde e irnos moviendo hacia la derecha
    #Tomamos puntos en x que esten igualmente espaciados para solo hacer el shooting en y.
    p0z = 0
    aprox = 0.5
    r00 = rmax0
    p0x = xval
    p0z = zval
    j = 0
    while j < n:
        #Primero obtenemos la solución no refinada y una masa aporximada
        p0y = yval + np.random.random()*yrange
        Ini0 = [p0x,0, p0y, 0, p0z, 0, 0, 0, 0]
        U0x, U0y, U0z, rTmax, *_ = Shoot_Proca1(Ini0, Uxint, Uyint, Uzint, r00,LambT=LambT, nodos=nodos, lim=1e-8,info=False, klim=1500, outval=10, 
                                                            delta=0.2)
        rmin, rmax = 0, rTmax
        Nptos = 2000; rspan = np.linspace(rmin, rmax, Nptos); arg = [LambT]
        met = 'RK45'; Rtol = 1e-09; Atol = 1e-10
        U0 = [p0x, 0, p0y, 0, p0z, 0, U0x, 0, U0y, 0, U0z, 0]
        sol = solve_ivp(systemProca, [rmin, rmax], U0, t_eval=rspan,
                         args=(arg,), method=met, rtol=Rtol, atol=Atol)
        rad = sol.t
        px, ux = sol.y[0], sol.y[6][0]
        py, uy = sol.y[2], sol.y[8][0]
        pz, uz = sol.y[4], sol.y[10][0]
        print("no refinado")
        U000 = [p0x,0,p0y,0,p0z, 0,U0x, 0,U0y,0,U0z,0]
        plotPerf(U000,rmax,0)
        # Definición del rango para el refinamiento, lo dividimos en tres partes
        for i in range(div):   
            rmin, rmax = 0, rTmax + rext0/div
            limit = [rmin, rmax]
             # Variables iniciales y semillas no refinadas
            p1x, p1y, p1z, u1x, u1y, u1z = 0,0,0,0,0,0
            uk = [U0x, U0y, U0z]
            # Configuración inicial del vector V0 y sus derivadas
            V0X = [p0x, 0, p0y, 0, p0z, 0, None, u1y, None, u1y, None, u1z]
            V0Duk = np.zeros(12 * 3)
            V0Duk[6] = 1  # Derivadas no triviales
            V0Duk[20] = 1
            V0Duk[-2] = 1
            V0 = np.concatenate((V0X, V0Duk))
    
            # Ajuste de las posiciones para las constantes ck
            indck = np.concatenate(([False, False, False, False, False, False, True, False, True, False, True, False], [False]*36))
            V0[indck] = uk
    
            # Índices de las variables para resolver y sus derivadas
            indXc = np.array([False] * 48)
            indXc[:5:2] = True  # Variables seleccionadas

            inddXc = np.array([False] * 48)
            inddXc[12::12] = True
            inddXc[14::12] = True
            inddXc[16::12] = True

            # Refinamiento usando fitting
            V0fit = fitting(
                system_refinado, V0, indck, indXc, [0,0,0], inddXc,
                limit, argf=LambT, tol=1e-5,npt = 1000, klim = 12000
                )
            rTmax = rmax
            U0 = V0fit[:12]
            U0x, U0y, U0z = U0[6], U0[8], U0[10]
            print(f"extension {i+1} completada")
        # Cálculo de la masa y energía total
        metodo = 'RK45'
        Rtol, Atol = 1e-9, 1e-10
        Nptos = 2000
        rspan = np.linspace(rmin, rmax, Nptos)
        U0 = V0fit[:12]
    
        sol = solve_ivp(
        systemProca, [rmin, rmax], U0, t_eval=rspan,
                args=([LambT],), method=metodo, rtol=Rtol, atol=Atol
                )
        rad = sol.t
        px, ux = sol.y[0], sol.y[6][0]
        py, uy = sol.y[2], sol.y[8][0]
        pz, uz = sol.y[4], sol.y[10][0]
        #Calculamos la masa
        #Esto es para recortar px porque hay algunas veces que py se ve bien pero px se empieza a levantar pero su contribucion claramente deberia ser 0
        if recx:
            rec = int(rext0*Nptos/rmax)
            px_r = px[:-rec]
            rad_r = rad[:-rec]
        else:
            px_r = px
            rad_r = rad
        if recy:
            rec = int(recorte*Nptos/rmax)
            py_r = py[:-rec]
            rad_y = rad[:-rec]
        else:
            py_r = py
            rad_y = rad
        if recz:
            rec = int(recorte*Nptos/rmax)
            pz_r = pz[:-rec]
            rad_z = rad[:-rec]
        else:
            pz_r = pz
            rad_z = rad
        Mx = Calc_Masa(rad_r, px_r**2)
        My = Calc_Masa(rad_y, py_r**2)
        Mz = Calc_Masa(rad_z, pz_r**2)
        if (sz==False):
            Mz = 0
        if (sx==False):
            Mx = 0
        if (sy==False):
             My = 0
        masaT = 4*np.pi*(Mx+My+Mz)
        auxx = [U0[6], U0[8], U0[10], rmax, [U0[0], U0[2], U0[5]],LambT]
            #Graficamos para asegurar que las soluciones sean validas
        U00 = [auxx[4][0],0,auxx[4][1],0,auxx[4][2], 0,auxx[0], 0,auxx[1],0,auxx[2],0]
        print(auxx)
        plotPerf(U00,auxx[3],0,px=False)
        print(masaT)
        j +=1
        

def shoot_random_base(n,xval=1,yval=1, yrange=0.1, Uxint=[0.1,2], Uyint=[0.1,2], Uzint=[0.1,2], rmax0=25, LambT=0, nodos=[0,1,0],sx=True,sy=True,sz=False):
    #Lo ideal es empezar desde la izquierde e irnos moviendo hacia la derecha
    #Tomamos puntos en x que esten igualmente espaciados para solo hacer el shooting en y.
    p0z = 0
    aprox = 0.5
    r00 = rmax0
    p0x = xval
    j = 0
    while j < n:
        #Primero obtenemos la solución no refinada y una masa aporximada
        p0y = yval + np.random.random()*yrange
        Ini0 = [p0x,0, p0y, 0, p0z, 0, 0, 0, 0]
        U0x, U0y, U0z, rTmax, *_ = Shoot_Proca1(Ini0, Uxint, Uyint, Uzint, r00,LambT=LambT, nodos=nodos, lim=1e-8,info=False, klim=1500, outval=10, 
                                                          delta=0.2)
        rmin, rmax = 0, rTmax
        Nptos = 2000; rspan = np.linspace(rmin, rmax, Nptos); arg = [LambT]
        met = 'RK45'; Rtol = 1e-09; Atol = 1e-10
        U0 = [p0x, 0, p0y, 0, p0z, 0, U0x, 0, U0y, 0, U0z, 0]
        sol = solve_ivp(systemProca, [rmin, rmax], U0, t_eval=rspan,
                         args=(arg,), method=met, rtol=Rtol, atol=Atol)
        rad = sol.t
        px, ux = sol.y[0], sol.y[6][0]
        py, uy = sol.y[2], sol.y[8][0]
        pz, uz = sol.y[4], sol.y[10][0]
        print("no refinado")
        U000 = [p0x,0,p0y,0,p0z, 0,U0x, 0,U0y,0,U0z,0]
        plotPerf(U000,rmax,0,px = False)
        print(U000)
        j +=1
             
#Esta función es para calcular las configuraciones estacionarias
def shoot_estacionario(masa, Uxint, Uyint, Uzint, rmax0, LambT, nodos,sx=False,sy=True,sz=False, tol = 0.05,kmax=20):
    #Lo ideal es empezar desde la izquierde e irnos moviendo hacia la derecha
    #Tomamos puntos en x que esten igualmente espaciados para solo hacer el shooting en y.
    p0z = 0
    ymin = 0
    aprox = 0.5
    config = []
    p0x = 0
    #Primero obtenemos la solución no refinada y una masa aporximada
    k = 0
    ymin = 0
    ymax = 0.1
    while True:
        p0y = shoot(ymin,ymax)
        Ini0 = [p0x,0, p0y, 0, p0z, 0, 0, 0, 0]
        U0x, U0y, U0z, rTmax, *_ = Shoot_Proca1(Ini0, Uxint, Uyint, Uzint, rmax0,LambT=LambT, nodos=nodos, lim=1e-8,info=False, klim=1500, outval=10, 
                                                          delta=0.2)
        rmin, rmax = 0, rTmax
        Nptos = 1000; rspan = np.linspace(rmin, rmax, Nptos); arg = [LambT]
        met = 'RK45'; Rtol = 1e-09; Atol = 1e-10
        U0 = [p0x, 0, p0y, 0, p0z, 0, U0x, 0, U0y, 0, U0z, 0]
        sol = solve_ivp(systemProca, [rmin, rmax], U0, t_eval=rspan,
                     args=(arg,), method=met, rtol=Rtol, atol=Atol)
        rad = sol.t
        px, ux = sol.y[0], sol.y[6][0]
        py, uy = sol.y[2], sol.y[8][0]
        pz, uz = sol.y[4], sol.y[10][0]
        # Definición del rango para el refinamiento
        rmin, rmax = 0, rTmax + 8.5
        limit = [rmin, rmax]
        # Variables iniciales y semillas no refinadas
        p1x, p1y, p1z, u1x, u1y, u1z = 0,0,0,0,0,0
        uk = [U0x, U0y, U0z]
        # Configuración inicial del vector V0 y sus derivadas
        V0X = [p0x, 0, p0y, 0, p0z, 0, None, u1y, None, u1y, None, u1z]
        V0Duk = np.zeros(12 * 3)
        V0Duk[6] = 1  # Derivadas no triviales
        V0Duk[20] = 1
        V0Duk[-2] = 1
        V0 = np.concatenate((V0X, V0Duk))
    
        # Ajuste de las posiciones para las constantes ck
        indck = np.concatenate(([False, False, False, False, False, False, True, False, True, False, True, False], [False]*36))
        V0[indck] = uk
    
        # Índices de las variables para resolver y sus derivadas
        indXc = np.array([False] * 48)
        indXc[:5:2] = True  # Variables seleccionadas

        inddXc = np.array([False] * 48)
        inddXc[12::12] = True
        inddXc[14::12] = True
        inddXc[16::12] = True
    
        # Refinamiento usando fitting
        V0fit = fitting(
                system_refinado, V0, indck, indXc, [0, 0, 0], inddXc,
                limit, argf=LambT, tol=1e-10
                )
        # Cálculo de la masa y energía total
        metodo = 'RK45'    
        Rtol, Atol = 1e-9, 1e-10
        Nptos = 500
        rspan = np.linspace(rmin, rmax, Nptos)
        U0 = V0fit[:12]
    
        sol = solve_ivp(
        systemProca, [rmin, rmax], U0, t_eval=rspan,
                args=([LambT],), method=metodo, rtol=Rtol, atol=Atol
                )
        rad = sol.t
        px, ux = sol.y[0], sol.y[6][0]
        py, uy = sol.y[2], sol.y[8][0]
        pz, uz = sol.y[4], sol.y[10][0]
        #Calculamos la masa
        Mx = Calc_Masa(rad, px**2)
        My = Calc_Masa(rad, py**2)
        Mz = Calc_Masa(rad, pz**2)
        if (sz==False):
            Mz = 0
        if (sx==False):
            Mx = 0
        if (sy==False):
            My = 0
        masaT = 4*np.pi*(Mx+My+Mz)
        if (abs(masaT-masa)<tol):
            #ux,uy,uz,rmax,[p0],lambT
            config.append([U0[6], U0[8], U0[10], rmax, [U0[0], U0[2], U0[5]],LambT])
            print(f"Primer estado excitado encontrado con masa {masaT}:")
            print(f"El valor de sigma_y es: {p0y}")
            print(config)
            break
        if(k == kmax):
            print("numero máximo de iteraciones")
            break
        if (masaT > masa + tol):
            ymax = p0y
        elif(masaT < masa - tol):
            ymin = p0y
        print(masaT)
    return config, p0y
##################################################################
def shoot_estacionario_z(masa,xmin0,xmax0,z,ymin0,ymax0, Uxint, Uyint, Uzint, rmax0, LambT, nodos0,sx=True,sy=True,sz=True, tol = 0.05,kmax=20,rext=10,x0 = True, y0 = False,z0=False,recx=False,recy=False,recz=False,recorte=0):
    #Lo ideal es empezar desde la izquierde e irnos moviendo hacia la derecha
    #Tomamos puntos en x que esten igualmente espaciados para solo hacer el shooting en y.
    p0z = z
    ymin = ymin0
    aprox = 0.5
    config = []
    ymax = ymax0
    #Primero obtenemos la solución no refinada y una masa aporximada
    k = 0
    xmin = xmin0
    xmax = xmax0
    while True:
        if x0:
            p0x = shoot(xmin,xmax)
        else:
            p0x = xmin
        if y0:
            p0y = shoot(ymin,ymax)
        else:
            p0y = ymin
        Ini0 = [p0x,0, p0y, 0, p0z, 0, 0, 0, 0]
        U0x, U0y, U0z, rTmax, *_ = Shoot_Proca1(Ini0, Uxint, Uyint, Uzint, rmax0,LambT=LambT, nodos=nodos0, lim=1e-8,info=False, klim=1500, outval=10, 
                                                          delta=0.2)
        rmin, rmax = 0, rTmax
        Nptos = 1000; rspan = np.linspace(rmin, rmax, Nptos); arg = [LambT]
        met = 'RK45'; Rtol = 1e-09; Atol = 1e-10
        U0 = [p0x, 0, p0y, 0, p0z, 0, U0x, 0, U0y, 0, U0z, 0]
        sol = solve_ivp(systemProca, [rmin, rmax], U0, t_eval=rspan,
                     args=(arg,), method=met, rtol=Rtol, atol=Atol)
        rad = sol.t
        px, ux = sol.y[0], sol.y[6][0]
        py, uy = sol.y[2], sol.y[8][0]
        pz, uz = sol.y[4], sol.y[10][0]
        U000 = [p0x,0,p0y,0,p0z, 0,U0x, 0,U0y,0,U0z,0]
        print("no refinado")
        plotPerf(U000,rmax,0)
        # Definición del rango para el refinamiento
        rmin, rmax = 0, rTmax + rext
        limit = [rmin, rmax]
        # Variables iniciales y semillas no refinadas
        p1x, p1y, p1z, u1x, u1y, u1z = 0,0,0,0,0,0
        uk = [U0x, U0y, U0z]
        # Configuración inicial del vector V0 y sus derivadas
        V0X = [p0x, p1x, p0y, p1y, p0z, p1z, None, u1y, None, u1y, None, u1z]
        V0Duk = np.zeros(12 * 3)
        V0Duk[6] = 1  # Derivadas no triviales
        V0Duk[20] = 1
        V0Duk[-2] = 1
        V0 = np.concatenate((V0X, V0Duk))
    
        # Ajuste de las posiciones para las constantes ck
        indck = np.concatenate(([False, False, False, False, False, False, True, False, True, False, True, False], [False]*36))
        V0[indck] = uk
    
        # Índices de las variables para resolver y sus derivadas
        indXc = np.array([False] * 48)
        indXc[:5:2] = True  # Variables seleccionadas

        inddXc = np.array([False] * 48)
        inddXc[12::12] = True
        inddXc[14::12] = True
        inddXc[16::12] = True
    
        # Refinamiento usando fitting
        V0fit = fitting(
                system_refinado, V0, indck, indXc, [0, 0, 0], inddXc,
                limit, argf=LambT, tol=1e-5
                )
        # Cálculo de la masa y energía total
        metodo = 'RK45'    
        Rtol, Atol = 1e-9, 1e-10
        Nptos = 500
        rspan = np.linspace(rmin, rmax, Nptos)
        U0 = V0fit[:12]
    
        sol = solve_ivp(
        systemProca, [rmin, rmax], U0, t_eval=rspan,
                args=([LambT],), method=metodo, rtol=Rtol, atol=Atol
                )
        rad = sol.t
        px, ux = sol.y[0], sol.y[6][0]
        py, uy = sol.y[2], sol.y[8][0]
        pz, uz = sol.y[4], sol.y[10][0]
        #Calculamos la masa
        if(recx):
            rec = int(rext*Nptos/rmax)
            px = px[:-rec]
            radx = rad[:-rec]
        else:
            radx = rad
        if(recy):
            rec = int(recorte*Nptos/rmax)
            py = py[:-rec]
            rady = rad[:-rec]
        else:
            rady=rad
        if(recz):
            rec = int(recorte*Nptos/rmax)
            pz = pz[:-rec]
            radz = rad[:-rec]
        else:
            radz=rad
        Mx = Calc_Masa(radx, px**2)
        My = Calc_Masa(rady, py**2)
        Mz = Calc_Masa(radz, pz**2)
        if (sz==False):
            Mz = 0
        if (sx==False):
            Mx = 0
        if (sy==False):
            My = 0
        masaT = 4*np.pi*(Mx+My+Mz)
        if (abs(masaT-masa)<tol):
            #ux,uy,uz,rmax,[p0],lambT
            config.append([U0[6], U0[8], U0[10], rmax, [U0[0], U0[2], U0[5]],LambT])
            print(f"Primer estado excitado encontrado con masa {masaT}:")
            print(f"El valor de sigma_x es: {p0x}")
            auxx = [U0[6], U0[8], U0[10], rmax, [U0[0], U0[2], U0[4]],LambT]
            U00 = [auxx[4][0],0,auxx[4][1],0,auxx[4][2], 0,auxx[0], 0,auxx[1],0,auxx[2],0]
            plotPerf(U00,auxx[3],0,px=False,py=False, pz=z0)
            plotPerf(U00,auxx[3],0,px=x0,py=y0, pz=z0)
            print(config)
            break
        if(k == kmax):
            print("numero máximo de iteraciones")
            break
        if x0:
            if (masaT > masa + tol):
                xmax = p0x
            elif(masaT < masa - tol):
                xmin= p0x
        if y0:
            if (masaT > masa + tol):
                ymax = p0y
            elif(masaT < masa - tol):
                ymin = p0y
        auxx = [U0[6], U0[8], U0[10], rmax, [U0[0], U0[2], U0[4]],LambT]
        U00 = [auxx[4][0],0,auxx[4][1],0,auxx[4][2], 0,auxx[0], 0,auxx[1],0,auxx[2],0]
        print(auxx)
        #plotPerf(U00,auxx[3],0,px=False,py=True, pz=True)
        plotPerf(U00,auxx[3]-recorte,0,px=False,py=False, pz=z0)
        plotPerf(U00,auxx[3],0,px=x0,py=y0, pz=z0)
        print(masaT)
    return config
#shot con z encendido
def shoot_random(n,xval=1,yval=1,zval=0.1, yrange=0.1, Uxint=[0.1,2], Uyint=[0.1,2], Uzint=[0.1,2], rmax0=25, LambT=0, nodos=[0,1,0],sx=True,sy=True,sz=False, tol = 0.05,estacionario = False,rext0 = 6):
    #Lo ideal es empezar desde la izquierde e irnos moviendo hacia la derecha
    #Tomamos puntos en x que esten igualmente espaciados para solo hacer el shooting en y.
    p0z = 0
    aprox = 0.5
    r00 = rmax0
    p0x = xval
    j = 0
    while j < n:
        #Primero obtenemos la solución no refinada y una masa aporximada
        p0y = yval + np.random.random()*yrange
        Ini0 = [p0x,0, p0y, 0, p0z, 0, 0, 0, 0]
        U0x, U0y, U0z, rTmax, *_ = Shoot_Proca1(Ini0, Uxint, Uyint, Uzint, r00,LambT=LambT, nodos=nodos, lim=1e-8,info=False, klim=1500, outval=10, 
                                                            delta=0.2)
        rmin, rmax = 0, rTmax
        Nptos = 2000; rspan = np.linspace(rmin, rmax, Nptos); arg = [LambT]
        met = 'RK45'; Rtol = 1e-09; Atol = 1e-10
        U0 = [p0x, 0, p0y, 0, p0z, 0, U0x, 0, U0y, 0, U0z, 0]
        sol = solve_ivp(systemProca, [rmin, rmax], U0, t_eval=rspan,
                         args=(arg,), method=met, rtol=Rtol, atol=Atol)
        rad = sol.t
        px, ux = sol.y[0], sol.y[6][0]
        py, uy = sol.y[2], sol.y[8][0]
        pz, uz = sol.y[4], sol.y[10][0]
        print("no refinado")
        U000 = [p0x,0,p0y,0,p0z, 0,U0x, 0,U0y,0,U0z,0]
        plotPerf(U000,rmax,0)
        # Definición del rango para el refinamiento
        rmin, rmax = 0, rTmax + rext0
        limit = [rmin, rmax]
         # Variables iniciales y semillas no refinadas
        p1x, p1y, p1z, u1x, u1y, u1z = 0,0,0,0,0,0
        uk = [U0x, U0y, U0z]
        # Configuración inicial del vector V0 y sus derivadas
        V0X = [p0x, 0, p0y, 0, p0z, 0, None, u1y, None, u1y, None, u1z]
        V0Duk = np.zeros(12 * 3)
        V0Duk[6] = 1  # Derivadas no triviales
        V0Duk[20] = 1
        V0Duk[-2] = 1
        V0 = np.concatenate((V0X, V0Duk))
    
        # Ajuste de las posiciones para las constantes ck
        indck = np.concatenate(([False, False, False, False, False, False, True, False, True, False, True, False], [False]*36))
        V0[indck] = uk
    
            # Índices de las variables para resolver y sus derivadas
        indXc = np.array([False] * 48)
        indXc[:5:2] = True  # Variables seleccionadas

        inddXc = np.array([False] * 48)
        inddXc[12::12] = True
        inddXc[14::12] = True
        inddXc[16::12] = True

        # Refinamiento usando fitting
        V0fit = fitting(
            system_refinado, V0, indck, indXc, [0,0,0], inddXc,
            limit, argf=LambT, tol=1e-7
            )
        # Cálculo de la masa y energía total
        metodo = 'RK45'
        Rtol, Atol = 1e-9, 1e-10
        Nptos = 2000
        rspan = np.linspace(rmin, rmax, Nptos)
        U0 = V0fit[:12]
    
        sol = solve_ivp(
        systemProca, [rmin, rmax], U0, t_eval=rspan,
                args=([LambT],), method=metodo, rtol=Rtol, atol=Atol
                )
        rad = sol.t
        px, ux = sol.y[0], sol.y[6][0]
        py, uy = sol.y[2], sol.y[8][0]
        pz, uz = sol.y[4], sol.y[10][0]
        #Calculamos la masa
        #Esto es para recortar px porque hay algunas veces que py se ve bien pero px se empieza a levantar pero su contribucion claramente deberia ser 0
        rec = int(rext0*Nptos/rmax)
        px_r = px[:-rec]
        rad_r = rad[:-rec]
        Mx = Calc_Masa(rad_r, px_r**2)
        My = Calc_Masa(rad, py**2)
        Mz = Calc_Masa(rad, pz**2)
        if (sz==False):
            Mz = 0
        if (sx==False):
            Mx = 0
        if (sy==False):
             My = 0
        masaT = 4*np.pi*(Mx+My+Mz)
        auxx = [U0[6], U0[8], U0[10], rmax, [U0[0], U0[2], U0[5]],LambT]
            #Graficamos para asegurar que las soluciones sean validas
        U00 = [auxx[4][0],0,auxx[4][1],0,auxx[4][2], 0,auxx[0], 0,auxx[1],0,auxx[2],0]
        print(auxx)
        plotPerf(U00,auxx[3],0,px=False)
        print(masaT)
        j +=1
#Computo de masa para el shooting sin los calculos de energia
def Calc_Masa(r, sigtot):
    sigF = interp1d(r, sigtot, kind='quadratic') #Primero obtenemos sigma por medio de una interpolación 
    def sigr2(r):
        return r**2*sigF(r)
    rmin = r[0]
    rfin = r[-1]
    #Integramos usando quad
    Mas = quad(sigr2, rmin, rfin)[0]  # masa: c*hb/(G*m*Lambda^(1/2))
    return Mas
#FORMATO Y PLOTING
def Plot_paper(datosSig0EnMas, cmap_colors=['#f0784d', '#2681ab'], colormap_resolution=50,xmax=1,ymax=1):
    # Datos
    mass = datosSig0EnMas[0, 0]  # La masa es la misma para todas las configuraciones
    Et = datosSig0EnMas[:, 1]
    sigx = datosSig0EnMas[:, 2]
    sigy = datosSig0EnMas[:, 3]

    # Creando barra de colores normalizada
    cmap = LinearSegmentedColormap.from_list('', cmap_colors)
    norm = mpl.colors.Normalize(vmin=min(Et), vmax=max(Et)) 
    sm = ScalarMappable(norm=norm, cmap=cmap)

    # Interpolando la curva
    fun = interp1d(sigx, sigy, kind="quadratic")
    sval = np.linspace(sigx[0], sigx[-1], colormap_resolution)

    # Crear figura
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4))

    # Colorear línea interpolada
    color = np.linspace(0, 2, sval.size)
    colored_line(sval, fun(sval), color, ax, linewidth=1, cmap=cmap)

    # Puntos intermedios con colores basados en Et
    ax.scatter(sigx[1:-1], sigy[1:-1], c=Et[1:-1], vmin=min(Et[1:-1]), vmax=max(Et[1:-1]), 
               s=25, cmap=cmap, zorder=100)

    # Configuraciones inicial y final
    ax.plot(sigx[-1], sigy[-1], ls='', marker='*', markersize=8, c=cmap_colors[1], 
            label=r'stationary (linear) $1$st-excited state')
    ax.plot(sigx[0], sigy[0], ls='', marker='*', markersize=8, c=cmap_colors[0], 
            label=r'stationary (linear) ground state')

    # Texto
    ax.text(x=0.005, y= ymax, s=r'$N=%d$' % mass, fontsize='small')

    # Límites y etiquetas
    ax.set_xlabel(r'$\sigma_{x 0}$')
    ax.set_ylabel(r'$\sigma_{y 0}$')
    ax.set_xlim(-0.03, xmax)
    ax.set_ylim(-0.01, ymax)

    # Barra de color
    cbar = fig.colorbar(sm, fraction=0.05, pad=0.02, ax=ax, aspect=20, location='right')
    cbar.ax.set_title(r'$\mathcal{E}$', fontsize=14)

    # Leyenda
    ax.legend(loc=(0, 0.02), frameon=False, fontsize=12.5, handletextpad=0.3)
    plt.savefig('Plot 0,1,0.png', dpi=300, bbox_inches='tight')
    plt.show()
#Aqui agregué el plot del paper, para tener todo más organizado
def Plot_paper_multi(data_list, labels=None,cmap_colors=['#f0784d', '#2681ab', '#7f00ff'], colormap_resolution=50, markers=['*', '+', 'o']):
    # Crear figura
    #['#f0784d', '#2681ab']
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
    EMax = data_list[0][:, 1]
    EMin = data_list[1][:, 1]
    # Iterar sobre cada conjunto de datos en data_list
    for i, (datosSig0EnMas, label, marker) in enumerate(zip(data_list, labels, markers)):
        # Datos
        mass = datosSig0EnMas[0, 0]  # La masa es la misma para todas las configuraciones
        Et = datosSig0EnMas[:, 1]
        sigx = datosSig0EnMas[:, 2]
        sigy = datosSig0EnMas[:, 3]

        # Crear barra de colores normalizada para la curva actual
        cmap = LinearSegmentedColormap.from_list('', cmap_colors)
        norm = mpl.colors.Normalize(vmin=min(EMin), vmax=max(EMax))
        #norm = mpl.colors.Normalize(vmin=min(Et), vmax=max(Et))
        sm = ScalarMappable(norm=norm, cmap=cmap)

        # Interpolación
        fun = interp1d(sigx, sigy, kind="quadratic")
        sval = np.linspace(sigx[0], sigx[-1], colormap_resolution)

        # Colorear línea interpolada
        color = np.linspace(0, 2, sval.size)
        colored_line(sval, fun(sval), color, ax, linewidth=1, cmap=cmap)

        # Puntos intermedios con colores basados en Et
        #ax.scatter(sigx[1:-1], sigy[1:-1], c=Et[1:-1], vmin=min(Et[1:-1]), vmax=max(Et[1:-1]), 
                   #s=25, cmap=cmap, zorder=100)

        # Configuraciones inicial y final con marcadores únicos
        ax.plot(sigx[0], sigy[0], ls='', marker=marker, markersize=8, c=cmap_colors[0])
        ax.plot(sigx[-1], sigy[-1], ls='', marker=marker, markersize=8, c=cmap_colors[-1],label=f'{label}')

    # Límites y etiquetas
    ax.text(x=0.005, y=0.30, s=r'$N=%d$' % mass, fontsize='small')
    ax.set_xlabel(r'$\sigma_{x 0}$')
    ax.set_ylabel(r'$\sigma_{y 0}$')
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.01, 0.3)

    # Barra de color común
    cbar = fig.colorbar(sm, fraction=0.05, pad=0.02, ax=ax, aspect=20, location='right')
    cbar.ax.set_title(r'$\mathcal{E}$', fontsize=14)

    # Leyenda
    ax.legend(loc='upper right', frameon=True, fontsize=10, title='Configuration')
    plt.savefig('(0,1,0),(0,2,0),(0,3,0),orig', dpi=300, bbox_inches='tight')
    plt.show()

#def colored_line3D(x, y, z, color_values,norm, ax, linewidth=2, cmap=cm.viridis):
 #   points = np.array([x, y, z]).T.reshape(-1, 1, 3)
  #  segments = np.concatenate([points[:-1], points[1:]], axis=1)
   # default_kwargs = {"capstyle": "butt"}
   # default_kwargs.update(lc_kwargs)
   # colors = cmap(norm(color_values))
   # lc = Line3DCollection(segments, cmap=cmap, norm=norm, linewidth=linewidth)
    #lc.set_array(color_values)
    #ax.add_collection(lc)"



def Plot_3D(data_list, labels=None,cmap_colors=['#f0784d', '#2681ab', '#7f00ff'], colormap_resolution=50, markers=['*', '+', 'o'],z_values=[0,0,0]):
    # Crear figura
    #['#f0784d', '#2681ab']
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    global_min = np.inf
    global_max = -np.inf

    # Calcular los valores mínimos y máximos de Et para todas las curvas
    for datosSig0EnMas in data_list:
        Et = datosSig0EnMas[:, 1]
        global_min = min(global_min, np.min(Et))
        global_max = max(global_max, np.max(Et))
    # Iterar sobre cada conjunto de datos en data_list
    for i,(datosSig0EnMas, label, marker) in enumerate(zip(data_list, labels, markers)):
        # Datos
        mass = datosSig0EnMas[0, 0]  # La masa es la misma para todas las configuraciones
        Et = datosSig0EnMas[:, 1]
        sigx = datosSig0EnMas[:, 2]
        sigy = datosSig0EnMas[:, 3]
        sigz = datosSig0EnMas[:, 4]
        # Interpolación
        fun = interp1d(sigx, sigy, kind="quadratic")
        sval = np.linspace(sigx[0], sigx[-1], 2000)
        cmap = LinearSegmentedColormap.from_list('', [(0,'#c23838'),(0.25,'#f0784d'),(0.5,'#1dc4ab'),(1,'#2681ab')],N=2000)
        norm = Normalize(vmin=global_min, vmax=global_max)  # Normalización global
        sm = ScalarMappable(norm=norm, cmap=cmap)
        # Colorear línea interpolada
        if i == 0:
            z = np.zeros_like(sval)
            color = Et
            #colored_line3D(sval, fun(sval),z, color,ax, linewidth=1, cmap=cmap)
            #colored_line3D(sval, fun(sval),z, color,norm, ax, linewidth=1, cmap=cmap)
            ax.scatter(sigx[:],sigy[:],0,c=Et[:],norm=norm,s=25,cmap=cmap,zorder=100)
            ax.plot(sval, fun(sval),z,color='gray', lw=2, alpha=0.3, zorder=90,label=f'{label}')
            #ax.plot(sigx[0], sigy[0],0, ls='', marker=marker, markersize=8, c=cmap_colors[0])
            #ax.plot(sigx[-1], sigy[-1],0, ls='', marker=marker, markersize=8, c=cmap_colors[-1],label=f'{label}')
        elif i == 1:
            color = Et
            z = np.zeros_like(sval)
            ax.scatter(sigx[:],0,sigy[:],c=Et[:],norm=norm,s=25,cmap=cmap,zorder=100)
            ax.plot(sval, z,fun(sval),color='blue', lw=2, alpha=0.3, zorder=90,label=f'{label}')
            #colored_line3D(sval,z, fun(sval), color, ax, linewidth=1, cmap=cmap)
            #colored_line3D(sval,z, fun(sval), color,norm, ax, linewidth=1, cmap=cmap)
            #ax.plot(sigx[0],0, sigy[0], ls='', marker=marker, markersize=8, c=cmap_colors[0])
            #ax.plot(sigx[-1],0, sigy[-1], ls='', marker=marker, markersize=8, c=cmap_colors[-1],label=f'{label}')
        elif i == 2:
            color = Et
            z = np.zeros_like(sval)
            #colored_line3D(z,sval, fun(sval), color, ax, linewidth=1, cmap=cmap)
            #colored_line3D(z,sval, fun(sval), color,norm, ax, linewidth=1, cmap=cmap)
            ax.scatter(0,sigx[:],sigy[:],c=Et[:],norm=norm,s=25,cmap=cmap,zorder=100)
            ax.plot(z,sval, fun(sval),color='yellow', lw=2, alpha=0.3, zorder=90,label=f'{label}')
            #ax.plot(0,sigx[0], sigy[0], ls='', marker=marker, markersize=8, c=cmap_colors[0])
            #ax.plot(0,sigx[-1], sigy[-1], ls='', marker=marker, markersize=8, c=cmap_colors[-1],label=f'{label}')
        else:
            color = Et
            z = np.full_like(sval,z_values[i])
            #colored_line3D(z,sval, fun(sval), color, ax, linewidth=1, cmap=cmap)
            #colored_line3D(z,sval, fun(sval), color,norm, ax, linewidth=1, cmap=cmap)
            ax.scatter(sigx[:],sigy[:],z_values[i],c=Et[:],norm=norm,s=25,cmap=cmap,zorder=100)
            ax.plot(sval, fun(sval),z,color='gray', lw=2, alpha=0.3, zorder=90)
            #ax.plot(0,sigx[0], sigy[0], ls='', marker=marker, markersize=8, c=cmap_colors[0])
            #ax.plot(0,sigx[-1], sigy[-1], ls='', marker=marker, markersize=8, c=cmap_colors[-1],label=f'{label}')
    # Límites y etiquetas
    ax.set_xlabel(r'$\sigma_{x 0}$')
    ax.set_ylabel(r'$\sigma_{y 0}$')
    ax.set_zlabel(r'$\sigma_{z 0}$')
    ax.set_xlim(-0, 1)
    ax.set_ylim(-0,0.4)
    ax.set_zlim(0,0.25)
    ax.invert_yaxis()

    # Barra de color común
    cbar = fig.colorbar(sm, fraction=0.05, pad=0.02, ax=ax, aspect=20, location='right')
    cbar.ax.set_title(r'$\mathcal{E}$', fontsize=12)

    # Leyenda
    ax.legend(loc='upper right', frameon=True, fontsize=10, title='Configuration')
    plt.savefig('Superficie_configuraciones', dpi=300, bbox_inches='tight')
    plt.show()

def plotPerf(U0, rTmax, LambT, px=True, py=True, pz=True, lim=True):
    rmin, rmax = 0, rTmax
    Nptos = 500; rspan = np.linspace(rmin, rmax, Nptos); arg = [LambT]
    met = 'RK45'; Rtol = 1e-09; Atol = 1e-10
    sol2 = solve_ivp(systemProca, [rmin, rmax], U0, t_eval=rspan,
                     args=(arg,), method=met, rtol=Rtol, atol=Atol)
    
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4.5),
                       sharex=True, sharey=False,
                       gridspec_kw=dict(hspace=0.1, wspace=.15))

    if px:
        ax.plot(sol2.t, sol2.y[0], color='#1f77b4', label=r'$\sigma^{(0)}_x = %3.2f$'%sol2.y[0][0])
    
    if py:
        ax.plot(sol2.t, sol2.y[2], c='#ff7f0e', label=r'$\sigma^{(0)}_y= %3.2f$'%sol2.y[2][0])
    
    if pz:
        ax.plot(sol2.t, sol2.y[4], c='k', label=r'$\sigma^{(0)}_z= %3.2f$'%sol2.y[4][0])

    ax.legend(frameon=False)
    if px and pz and py:
        if lim:
            ax.set_ylim(np.min([sol2.y[0], sol2.y[2],sol2.y[4]])-abs(np.min([sol2.y[0], sol2.y[2],sol2.y[4]])/10), np.max([sol2.y[0][0], sol2.y[2][0],sol2.y[4][0]])+np.max([sol2.y[0][0], sol2.y[2][0], sol2.y[4][0]])/10)
        ax.set_xlim(0, sol2.t[-1])
    elif px and py:
        if lim:
            ax.set_ylim(np.min([sol2.y[0], sol2.y[2]])-abs(np.min([sol2.y[0], sol2.y[2]])/10), np.max([sol2.y[0][0], sol2.y[2][0]])+np.max([sol2.y[0][0], sol2.y[2][0]])/10)
        ax.set_xlim(0, sol2.t[-1])
    elif px and pz:
        if lim:
            ax.set_ylim(np.min([sol2.y[0], sol2.y[4]])-abs(np.min([sol2.y[0], sol2.y[4]])/10), np.max([sol2.y[0][0], sol2.y[4][0]])+np.max([sol2.y[0][0], sol2.y[4][0]])/10)
        ax.set_xlim(0, sol2.t[-1])    
    elif py and pz:
        if lim:
            ax.set_ylim(np.min([sol2.y[2], sol2.y[4]])-abs(np.min([sol2.y[2], sol2.y[4]])/10), np.max([sol2.y[2][0], sol2.y[4][0]])+np.max([sol2.y[2][0], sol2.y[4][0]])/10)
        ax.set_xlim(0, sol2.t[-1])   
    
        #ax.set_ylim(np.min([sol2.y[0], sol2.y[2]])-0.05, 0.1)  #np.max([sol2.y[0][0], sol2.y[2][0]])+0.05
        ax.set_xlim(0, sol2.t[-1])
    
    ax.hlines(y=0, xmin=0, xmax=sol2.t[-1], ls='--', lw=0.5, color='k')

    ax.set_ylabel(r'$\sigma^{(0)}_i$')
    ax.set_xlabel(r'$r$')
    plt.show()
    

# https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)
    

def general():
    """
    CONFIGURACIÓN GENERAL
    """
    # CONFIGURACIÓN GENERAL

    # Figura
    # mpl.rcParams['figure.dpi'] = 100 # figure dots per inch
    mpl.rcParams['figure.figsize'] = [12, 9]  # [4,3]
    mpl.rcParams['figure.facecolor'] = 'white'     # figure facecolor
    mpl.rcParams['figure.edgecolor'] = 'white'     # figure edgecolor
    mpl.rcParams['savefig.dpi'] = 100

    # estilo de linea y grosor
    mpl.rcParams['lines.linestyle'] = '-'
    mpl.rcParams['lines.linewidth'] = 2.

    # orden de los colores que usará
    mpl.rcParams['axes.prop_cycle'] = cycler('color',
                                             ['#1f77b4', '#ff7f0e', '#2ca02c',
                                              '#d62728', '#9467bd', '#8c564b',
                                              '#e377c2', '#7f7f7f', '#bcbd22',
                                              '#17becf'])
    # latex modo math
    # Should be: 'dejavusans' (default), 'dejavuserif', 'cm' (Computer Modern),
    #             'stix', 'stixsans' or 'custom'
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['mathtext.fallback'] = 'cm'

    # latex modo text
    # Should be: serif, sans-serif, cursive, fantasy, monospace
    mpl.rcParams['font.family'] = 'serif'
    cmfont = font_manager.FontProperties(fname=mpl.get_data_path()
                                         + '/fonts/ttf/cmr10.ttf')
    mpl.rcParams['font.serif'] = cmfont.get_name()
    mpl.rcParams['font.size'] = 16  # size of the text
      

    # Display axis spines, (muestra la linea de los marcos)
    mpl.rcParams['axes.spines.left'] = True
    mpl.rcParams['axes.spines.bottom'] = True
    mpl.rcParams['axes.spines.top'] = True
    mpl.rcParams['axes.spines.right'] = True

    # axes numbers, etc.
    # 'large' tamaño de los números de las x, y
    mpl.rcParams['xtick.labelsize'] = 13
    mpl.rcParams['ytick.labelsize'] = 13
    # direction: {in, out, inout} señalamiento de los ejes
    mpl.rcParams['xtick.direction'] = 'out'
    mpl.rcParams['ytick.direction'] = 'out'
    # draw ticks on the top side (dibujar las divisiones arriba)
    mpl.rcParams['xtick.top'] = False
    mpl.rcParams['ytick.right'] = False
    # draw label on the top/bottom
    mpl.rcParams['xtick.labeltop'] = False
    mpl.rcParams['xtick.labelbottom'] = True
    # draw x axis bottom/top major ticks
    mpl.rcParams['xtick.major.top'] = False
    mpl.rcParams['xtick.major.bottom'] = True
    # draw x axis bottom/top minor ticks
    mpl.rcParams['xtick.minor.top'] = False
    mpl.rcParams['xtick.minor.bottom'] = False

    # mpl.rcParams['xtick.minor.visible'] = True # visibility of minor ticks on x-axis

    # labels and title
    mpl.rcParams['axes.titlepad'] = 6.0  # pad between axes and title in points
    mpl.rcParams['axes.labelpad'] = 3.0  # 10.0     # space between label and axis
    mpl.rcParams['axes.labelweight'] = 'normal'  # weight (grosor) of the x and y labels
    mpl.rcParams['axes.labelcolor'] = 'black'
    mpl.rcParams['axes.unicode_minus'] = False  # use Unicode for the minus symbol
    mpl.rcParams['axes.linewidth'] = 1  # edge linewidth, grosor del marco

    mpl.rcParams['axes.titlesize'] = 24  # title size
    mpl.rcParams['axes.labelsize'] = 15  # label size
    # mpl.rcParams['lines.markersize'] = 10  # weight of the marker

    # Legend
    mpl.rcParams['legend.loc'] = 'best'
    mpl.rcParams['legend.frameon'] = True  # if True, draw the legend on a background patch
    mpl.rcParams['legend.framealpha'] = 0.19  # 0.8 legend patch transparency
    mpl.rcParams['legend.facecolor'] = 'inherit'  # inherit from axes.facecolor; or color spec
    mpl.rcParams['legend.edgecolor'] = 'inherit' # background patch boundary color
    mpl.rcParams['legend.fancybox'] = True  # if True, use a rounded box for the

    # mpl.rcParams['legend.numpoints'] = 1 # the number of marker points in the legend line
    # mpl.rcParams['legend.scatterpoints'] = 1 # number of scatter points
    # mpl.rcParams['legend.markerscale'] = 1.0 # the relative size of legend markers vs. original
    mpl.rcParams['legend.fontsize'] = 15  # 'medium' 'large'
    mpl.rcParams['legend.title_fontsize'] = 13  # 'xx-small'

    # Dimensions as fraction of fontsize:
    mpl.rcParams['legend.borderpad'] = 0.4  # border whitespace espacio de los bordes con respecto al texto
    mpl.rcParams['legend.labelspacing'] = 0.5  # the vertical space between the legend entries
    mpl.rcParams['legend.handlelength'] = 1.5  # the length of the legend lines defauld 2
    # mpl.rcParams['legend.handleheight'] = 0.7  # the height of the legend handle
    mpl.rcParams['legend.handletextpad'] = 0.8  # the space between the legend line and legend text
    # mpl.rcParams['legend.borderaxespad'] = 0.5  # the border between the axes and legend edge
    # mpl.rcParams['legend.columnspacing'] = 8.0  # column separation

# latex
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['pgf.rcfonts'] = False
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amssymb}'  # , \usepackage{txfonts}
    mpl.rcParams['pgf.preamble'] = r'\usepackage{amssymb}'

    return