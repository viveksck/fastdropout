function [f,g] = LinRegLossMCDropoutSample(w,X,y,ps, miniter, numiter)
[n,p]=size(X);
[is,js,vs] = find(X);

fs = zeros(numiter,1);
g = zeros(p, 1);
for i = 1:numiter
    Z = sparse(is,js,1*(rand(size(vs)) < ps), n, p, length(is));
    ZX = Z.*X;
    ZXw = (ZX)*w;
    
    fs(i) = (0.5)*(y-ZXw)'*(y-ZXw);

    g = g + (1 * ((ZX)'*(ZX))*w - (ZX)'*y);
    
    if i > miniter && (std(fs(1:i),1)/sqrt(i) < 1e-3*mean(fs(1:i)) )
        break
    end
end
g = g ./ numiter;
%fprintf('var of mean fs %f at i=%d\n', std(fs(1:i),1)/sqrt(i),i);
f = mean(fs(1:i));
end
