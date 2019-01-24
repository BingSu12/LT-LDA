function transMatrix = getTransChange(sigmaw,distb,downdim)

p = orth(sigmaw);
gama1 = p'*sigmaw*p;
w1 = p*(gama1^(-0.5));
s2 = w1'*distb*w1;
p2 = orth(s2);
gama2 = p2'*s2*p2;
rankdim = rank(gama2);
if rankdim<downdim
    disp('Too many dims have been perserved!');
end
a = zeros(1,rankdim);
for i=1:rankdim
    a(i)=-gama2(i,i);
end
[d,index]=sort(a);
q = [];
for i=1:downdim
    q = [q p2(:,index(i))];
end
transMatrix = w1*q;