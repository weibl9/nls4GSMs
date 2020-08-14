function Q = case2opt(xi,t,x,v)
%OPTFCN 'lsqnonlin' objective function
%   xi: parameter estimates [a b1 b2 eta]
%   t:  time instants nx1 column vector
%   x:  time-series observations
%   v:  exogenous varibale observations [v1 v2]

nobs = length(t);
h = [nan;diff(t)];
xfit = nan(nobs,1);

a = xi(1); b1 = xi(2); 
b2 = xi(3); eta = xi(4);

xfit(1) = eta;
for k=2:nobs
    tmp = 0;
    for i=2:k   % convolution 
        tmp=tmp+(exp(a*(t(1)-t(i-1)))*(b1*v(i-1,1)+b2*v(i-1,2) )+ ...
                 exp(a*(t(1)-t(i))  )*(b1*v(i,1)+b2*v(i,2)     ) )*h(i)/2;
    end
    xfit(k) = exp(a*(t(k)-t(1)) )*(eta+tmp );
end
err = x-xfit;

Q = err;
if ~isfinite(Q)
  Q = 1e+23;
end

end
















