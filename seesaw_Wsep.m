
function Qmax = seesaw_Wsep
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function performs a see-saw optimization procedure 
% to compute the maximum witness value Qmax = W_sep(D). 
% The results for Table I in the paper are provided as:
%
%        D    W_Cl      W_sep^{LB}     W_sep^{UB}       
% RES = [4    16384     16384.0000     16384; 
%        5    18048     19629.6336     20480; 
%        6    22016     23985.1201     24576; 
%        7    25856     27507.8035     28672;
%        8    32768     32767.9999     32768;
%        9    33280     35411.2734     36864;
%       10    35584     39460.0594     40960;
%       11    38656     42571.2328     45056;
%       12    44032     47442.2300     49152;
%       13    47360     50810.4552     53248;
%       14    54016     55697.7201     57344; 
%       15    57856     59583.9999     61440;  
%       16    65536     65536.0000     65536];
%
% Note: Due to normalization, all W values should be divided 
% by 4^8 in accordance with the witness formula in the paper.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Setting up the parameters:

% Dimension 
d = 8;

% Choose between real, complex, and classical state spaces:
% 0 = real valued, 1 = complex valued, 2 = classical dit case 
iscomp = 1;

switch iscomp
    case 0
        disp(['real valued case; d = ', num2str(d)]);
       
    case 1
        disp(['complex valued case; d = ', num2str(d)]);
        
    case 2
        disp(['classical dit case; d = ', num2str(d)]);
        
    otherwise
        disp('The iscomp value does not match any case');
        return;
end

% Define the witness matrix
T = 2*[1 1 1 1; 1 1 -1 -1; 1 -1 1 -1; 1 -1 -1 1];
F = kron(T,T);

% Define parameters for the see-saw iterations 
maxit = 1000;    % maximum number of iterations per run
maxrun = 100;    % number of see-saw runs
epsstop = 1e-9;  % convergence parameter 

[ds,dm]=size(F);

qmax = -1e+9;
for irun = 1:maxrun
disp(['iteration No: ', num2str(irun)]);
    
% Initialize random state preparations
switch iscomp
    case 0 % real-valued case
        psi = normc(randn(d,ds));
    case 1 % complex-valued case
        psi = randn(d,ds)+1i*randn(d,ds);
        for i = 1:ds
            psi(:,i) = psi(:,i)/norm(psi(:,i));
        end
    case 2 % classical dit case
        r = randi([1 d],1,ds);
        psi = zeros(d,ds);
        psi(sub2ind([d,ds],r,1:ds)) = 1;
end

% See-saw procedure
q = -1e+9;
qold = -2e+9;
epsi = 1e-12;
numit = 0;
while (q-qold > epsstop) && (numit <  maxit)
    numit = numit+1;
    qold = q;
    
    % Compute projectors tau_x corresponding to pure states v(:,x)
    tau=zeros(d,d,ds);
    for x=1:ds
        tau(:,:,x)=psi(:,x)*psi(:,x)';
    end
    
    % Compute optimal measurements C_z
    for z=1:dm
        Oz=zeros(d,d);
        for x=1:ds
            Oz=Oz+F(x,z)*tau(:,:,x);
        end
        Oz=(Oz+Oz')/2;
        [phi,mu]=eig(Oz);
        Cz(:,:,z)=phi*sign(mu+epsi*eye(d,d))*phi';
    end
    
    % Evaluate the coefficients s_z 
    sz = zeros(1,dm);
    for z=1:dm
        sumz = 0;
        for x=1:ds
            sumz=sumz+F(x,z)*psi(:,x)'*Cz(:,:,z)*psi(:,x);
        end
        sz(z) = sumz;
    end
  
    % Compute optimally prepared states rho_x
    q=0;
    for x=1:ds
        M=zeros(d,d);
        for z=1:dm
            M=M+sz(z)*F(x,z)*Cz(:,:,z);
        end
        M=(M+M')/2;
       [psivec,lambda]=eig(M);
        psi(:,x)=psivec(:,d);
        q=q+lambda(d,d);
    end
    % return to the beginning of the loop
end

if(q > qmax), qmax = q; end

format long g
fprintf('Actual qmax:  %f Best qmax:  %f\n', q, qmax);
format short

Qmax = qmax;
end