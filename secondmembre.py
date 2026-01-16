import equations
import numpy as np

def calcul_premier_second_membre_1(T,phi,j,dx,dy,dt,Prandt):
    (N_x, N_y) = np.shape(T)
    mat_Z=np.zeros(N_x-2)
    mat_Y=np.zeros(N_x-2)
    for i in range(2, N_x-1):
        
            alpha_i_j_y=phi((i,j+1))-phi((i,j-1))/(2*dy)
            Z=alpha_i_j_y*dt/(2*dx)
            
            alpha_i_j_x=phi((i+1,j))-phi((i-1,j))/(2*dx)
            Y=Z=alpha_i_j_x*dt/(2*dy)
            
            mat_Z((i-2))=Z
            mat_Y((i-2))=Y
            B_x=1/Prandt*dt/(dx**2)
            B_y=1/Prandt*dt/(dy**2)
            
            premier_membre=np.zeros((N_x-2, N_x-2))
            second_membre=np.zeros((N_x-2))
            for i in range(0,N_x-2):
                premier_membre((i,i))=1+2*B_x
                second_membre=-T((i))*2*B_y+T((i,j+1))*(B_y+Y((i)))+T((i,j-1))*(B_y-Y((i)))
                if i<=N_x-3:
                    premier_membre(((i,i+1)))=Z((i))-B_x
                    premier_membre(((i-1,i)))=-Z((i))-B_x
            
            return(premier_membre,second_membre)
        
        
        
def calcul_premier_second_membre_2(T,phi,i,dx,dy,dt,Prandt):
    (N_x, N_y) = np.shape(T)
    mat_Z=np.zeros(N_y-2)
    mat_Y=np.zeros(N_y-2)
    for j in range(2, N_y-1):
        
            alpha_i_j_x=-phi((i,j+1))-phi((i,j-1))/(2*dy)
            Z=alpha_i_j_x*dt/(2*dx)
            
            alpha_i_j_y=-phi((i+1,j))-phi((i-1,j))/(2*dx)
            Y=Z=alpha_i_j_y*dt/(2*dy)
            
            mat_Z((i-2))=Z
            mat_Y((i-2))=Y
            B_x=1/Prandt*dt/(dy**2)
            B_y=1/Prandt*dt/(dx**2)
            
            premier_membre=np.zeros((N_x-2, N_x-2))
            second_membre=np.zeros((N_x-2))
            for i in range(0,N_x-2):
                premier_membre((i,i))=1+2*B_x
                second_membre=-T((i))*2*B_y+T((i,j+1))*(B_y+Y((i)))+T((i,j-1))*(B_y-Y((i)))
                if i<=N_x-3:
                    premier_membre(((i,i+1)))=Z((i))-B_x
                    premier_membre(((i-1,i)))=-Z((i))-B_x
            
            return(premier_membre,second_membre)
                
                    
            
            
            
    
            
            
            
    
    

