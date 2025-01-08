clear
close all
clc
% 图像参数设定
imagesize=1024;
x=1:imagesize;
y=1:imagesize;
[X,Y]=meshgrid(x,y);
X=(X-imagesize/2)/(imagesize/2);
Y=(Y-imagesize/2)/(imagesize/2);
n=101;
for aa=n:n
%% 读取初始图片
filename=['testimgs\',num2str(aa),'.jpg'];
I=im2double(imread(filename));
PhaseDiffy=I*2*pi-pi;
% figure,imshow(PhaseDiffy,[])
% 生成标准背景图
filename=['testlabels\',num2str(aa),'.txt'];
p=readmatrix(filename);
m=p(1);
n=p(2);
am1=p(3);
rou1_1=p(4);
rou1_2=p(5);
xx=((X-m).^2)/rou1_1/rou1_1;
yy=((Y-n).^2)/rou1_2/rou1_2;
% phase_shearo=-1*(X-m)*am1/rou1_1/rou1_1.*exp(-(xx+yy)/2);
% phase_shearo_truth=angle(exp(1i*phase_shearo));
% phasepure=angle(exp(1i*(PhaseDiffy-phase_shearo_truth)));
% % figure,imshow(phasepure,[])
% fp=im2uint8(mat2gray(phasepure));
% filename=['testlabels\','f',num2str(aa),'.jpg'];
% imwrite(fp,filename)
%% 生成预测背景图
% filename=['testresult\',num2str(aa),'.txt'];
% p2=readmatrix(filename);
m=p2(1);
n=p2(2);
am1=p2(3);
rou1_1=p2(4);
rou1_2=p2(5);
xx=((X-m).^2)/rou1_1/rou1_1;
yy=((Y-n).^2)/rou1_2/rou1_2;
phase_shearo=-1*(X-m)*am1/rou1_1/rou1_1.*exp(-(xx+yy)/2);
phase_shearo_predict=angle(exp(1i*phase_shearo));
figure,imshow(phase_shearo_predict,[])
phasepredict=angle(exp(1i*(PhaseDiffy-phase_shearo_predict)));
figure,imshow(phasepredict,[])
% fp=im2uint8(mat2gray(phasepredict));
% filename=['testresult\','f',num2str(aa),'.jpg'];
% imwrite(fp,filename)
end