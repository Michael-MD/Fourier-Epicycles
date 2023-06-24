You can create your own compelx Fourier Series visualizations as shown here. You can make anything you want really, I originally got this idea from 3Blue1Brown's video <a href="https://www.youtube.com/watch?v=r6sGWTCMz2k">here </a>. Mathologer also casts this result as a question of paths traced by epicycles, the video can be found <a href="https://www.youtube.com/watch?v=qS4H6PEcCCA&t=656s">here</a>. For some cool examples see the examples section.

# 1. Examples

![alt text](example animations/ssdsd.)

# 2. Usage
To use this script simple open terminal window, navigate to the root directory where the code is located. For example, 
```
cd /path/to/root/
``` 
Then run the following line which takes in four arguments
```
python fourier_visualization path/to/img frac_points_used export_animation use_main_curve
```


# 3. Theory
The rich theory of Fourier Analysis has become one of the most utilized fields by the applied sciences. The theory is applicable to many classes of functions. Here we consider the complex Fourier Series which possess a quick numerical implementation named the FFT. A Fourier Series is the expansion of a sufficiently well-behaved function in a basis set of complex exponentials of varying frequencies. That is 

$$  f(n) =\frac{1}{N} \sum_{k=0}^{N-1} c_k \exp(j2\pi k n / N) $$

where N is the period of the signal in samples. The coefficients $c_k$ encode the "strength" of each frequency in the signal. If we look at this expansion in the complex plane then these are simply phasors tip-to-tip which are the circles the circles you will see in the demonstration below.  The radius of each circle is given by $|c_k|$ and the phase for a given n is given by the complex exponential offset by the phase of $c_k$.