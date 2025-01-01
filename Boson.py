#Librerias
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from cycler import cycler
from scipy.integrate import quad
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from matplotlib.collections import LineCollection

#Sistema de ecuaciones a resolver Para bosones 1D
def systemBos(r, cond, arg):
    """
    Sistema de ecuaciones
    [sigx, sigx', ux, ux'] -> [s0x, s1x,u0x, u1x]
    """
    p0x, p1x, u0x, u1x = cond
    LambT, = arg
    if r > 0:
        f0 = p1x
        f1 = LambT*p0x**3 - (2*p1x)/r - p0x*u0x
        f2 = u1x
        f3 = -p0x**2-(2*u1x)/r
    else:
        f0 = p1x
        f1 = (LambT*p0x**3 - p0x*u0x)/3
        f2 = u1x
        f3 = -p0x**2/3
        
    return [f0, f1, f2, f3]
#Shoting para una dimensión
def Bosshot(Ini0, Uintx,rmax, rmin=0, LambT=1, n=0, 
                     met='RK45', Rtol=1e-09, Atol=1e-10, lim=1e-6, info=False,
                     klim=500, outval=13, delta=0.4):
    p0x, p1x, u1x, = Ini0

    #p0Data = [p0x, p0y, p0z]

    Uminx, Umaxx = Uintx
    print(f'Encontrando un perfil para el caso unidimensional con: {n} nodos')

    # Los eventos que nos permiten contar los nodos.
    def Sigx(r, U, arg): return U[0] #sigma
    def dSigx(r, U, arg): return U[1] #la derivada
    Sigx.direction = 0; dSigx.direction = 0 #No importa si es por abajo o por arriba
    k = 0
    
    arg = [LambT]
    #Uintrs = np.array([[Uminx, Umaxx], [Uminy, Umaxy], [Uminz, Umaxz]])
    out = 0
    while True:
        u0 = shoot(Uminx,Umaxx)
        V0 = [p0x, p1x, u0, u1x] #actualizamos con el nuevo u0
        
        sol = solve_ivp(systemBos, [rmin, rmax], V0, events=(Sigx, dSigx),
                         args=(arg,), method=met,  rtol=Rtol, atol=Atol) #Resolvemos para luego volver a hacer el shooting.

        if info: #si se encontró solución con shooting correcto grafica#
            plt.plot(sol.t, sol.y[0], c='red')
            plt.xlim(0, 30)
            plt.ylim(-1.5, 1.5)
            plt.show()
  
        eventos = np.array([sol.t_events[0], sol.t_events[1]], dtype=object) #Agrupamos todos los eventos 
        New_Interv, rTemp = freq_shoot1D(eventos, n, u0, Uintx, rmax) #realizamos el shooting para sigma 
        
        if abs((New_Interv[1]-New_Interv[0])/2) <= lim:
            if info:
                print(out)
                print('Maxima precisión alcanzada: U0x = ', V0[2], 'radio', rTemp)
            
            if out==outval: #outval es el número máximo de veces que se puede extender el intervalo
                print('Maxima precisión alcanzada: U0x = ', V0[2], 'radio', rTemp)
                return V0[2], rTemp, sol.t_events[0]
            else:
                Uintx = [New_Interv[0]-delta, New_Interv[1]+delta] #No se encontró, por lo tanto se agranda el intervalo
                out += 1
        else:
            Uintx = New_Interv #Se actualiza el intervalo
            
        if info:
            print(Uintx)
        Uminx, Umaxx = Uintx
        if (shoot(Uminx,Umaxx) ==u0): #Si el shooting del intervalo anterior es igual al del nuevo intervalo, significa que ya acabamos
            print('Found: U0x = ', V0[2], 'radio', rTemp)
            return V0[2], rTemp, sol.t_events[0]
        
        if k==klim:
            print('loop limit reached')
            break
            
        k += 1
        

def shoot(imin, imax):
        return (imin+imax)/2

def freq_shoot1D(events, nodos, u0, New_Interv, rTemp):
    imin, imax = New_Interv
    if events[0].size == nodos and events[1].size == nodos+1:
        return [imin, imax], rTemp
    elif events[1].size > nodos+1:
        if events[0].size > nodos:  # dos veces por nodo
            imax = u0
            rTemp = events[0][-1]
        else:  # si pasa por cero más veces que 2*nodos se aumenta la w, sino se disminuye
            imin = u0
            rTemp = events[1][-1]
    elif events[1].size <= nodos+1:
        if events[0].size > nodos:  # dos veces por nodo
            imax = u0
            rTemp = events[0][-1]
        else:
            imin = u0
            rTemp = events[1][-1]
    return [imin, imax], rTemp


def general():
    """
    CONFIGURACIÓN GENERAL
    """
    # CONFIGURACIÓN GENERAL, esto solo lo copie y lo pegué, igual solo es para tener el mismo formato en las gráficas

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
###### PLOT
def plotPerfBos(U0, rTmax, LambT, px=True, lim=True):
    rmin, rmax = 0, rTmax
    Nptos = 500; rspan = np.linspace(rmin, rmax, Nptos); arg = [LambT]
    met = 'RK45'; Rtol = 1e-09; Atol = 1e-10
    sol2 = solve_ivp(systemBos, [rmin, rmax], U0, t_eval=rspan,
                     args=(arg,), method=met, rtol=Rtol, atol=Atol)
    
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4.5),
                       sharex=True, sharey=False,
                       gridspec_kw=dict(hspace=0.1, wspace=.15))

    
    ax.plot(sol2.t, sol2.y[0], color='#1f77b4', label=r'$\sigma^{(0)}_x = %3.2f$'%sol2.y[0][0])
    ax.legend(frameon=False)

    if lim:
        ax.set_ylim(np.min([sol2.y[0], sol2.y[2]])-abs(np.min([sol2.y[0], sol2.y[2]])/10), np.max([sol2.y[0][0], sol2.y[2][0]])+np.max([sol2.y[0][0], sol2.y[2][0]])/10)
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
    
