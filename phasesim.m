clear
close all
clc

imagesize=1024;
x=1:imagesize;
y=1:imagesize;
[X,Y]=meshgrid(x,y);
X=(X-imagesize/2)/(imagesize/2);
Y=(Y-imagesize/2)/(imagesize/2);

%% 高斯变形

num=1000;
for j=1:num
    am1=15+20*rand(1);  %am1=25+20*rand(1);
    rou1_1=0.65+0.65*rand(1); %rou1_1=0.5+0.6*rand(1);
    rou1_2=rou1_1*0.7+rou1_1*0.6*rand(1);
    m=0.3*rand(1)-0.15;
    n=0.3*rand(1)-0.15;
    xx=((X-m).^2)/rou1_1/rou1_1;
    yy=((Y-n).^2)/rou1_2/rou1_2;
    phase_shearo=-1*(X-m)*am1/rou1_1/rou1_1.*exp(-(xx+yy)/2);
    phase_shearo_bl=angle(exp(1i*phase_shearo));
    % figure,imshow(phase_shearo_bl,[])
    time=randi(5)-1;
    if time ~= 0
        for i=1:time
            am2=65+70*rand(1); %am2=75+70*rand(1)
            rou2_1=rand(1)*0.02+0.1;
            rou2_2=rou2_1*0.8+rou2_1*0.4*rand(1);
            M=1.2*rand(1)-0.6;
            N=1.2*rand(1)-0.6;
            xx=((X-M).^2)/rou2_1/rou2_1;
            yy=((Y-N).^2)/rou2_2/rou2_2;
            phase_shearo_ld=-1*(X-M)*am2.*exp(-(xx+yy)/2);
            phase_shearo=phase_shearo+phase_shearo_ld;
            % figure,imshow(angle(exp(1i*phase_shearo_ld)),[])
        end
    end
    % figure,mesh(phase_shearo)
    phase_shearo_mod=angle(exp(1i*phase_shearo));
    % figure,imshow(phase_shearo_mod,[])
    window=randi(5000)+2500;
    PhaseBefore=rand(imagesize,imagesize);
    PhaseBefore=PhaseBefore/(max(PhaseBefore(:)))*pi*4; %最大值=2pi*2
    PhaseAfter=phase_shearo+PhaseBefore;
    FreqLowPassFilter=X.^2+Y.^2<(imagesize/window)^2;
    ImageCaBefore=ifft2(ifftshift(fftshift(fft2(exp(1i*PhaseBefore))).*FreqLowPassFilter));%变形前，像面光场复振幅
    ImageCaAfter=ifft2(ifftshift(fftshift(fft2(exp(1i*PhaseAfter))).*FreqLowPassFilter));%变形后，像面光场复振幅
    PhaseDiffy=angle(ImageCaAfter.*conj(ImageCaBefore));
    % figure,imshow(PhaseDiffy,[])
    filename=['testimgs\',num2str(j),'.jpg'];
    PD=im2uint8(mat2gray(PhaseDiffy));
    imwrite(PD,filename);
    T=[m,n,am1,rou1_1,rou1_2];
    filename=['testlabels\',num2str(j),'.txt'];
    writematrix(T,filename)
end


