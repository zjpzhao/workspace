# Fault tolerance in distributed systems

``` ad-info
title: Metadata
- **CiteKey**: jaloteFaultToleranceDistributed1994
- **Type**: book
- **Author**: Jalote, Pankaj
- **Editor**: {{editor}}
- **Translator**: {{translator}}
- **Publisher**: PTR Prentice Hall
- **Location**: Englewood Cliffs, N.J
- **Series**: {{series}}
- **Series Number**: {{seriesNumber}}
- **Journal**: {{publicationTitle}}
- **Volume**: {{volume}}
- **Issue**: {{issue}}
- **Pages**: {{pages}}
- **Year**: 1994 
- **DOI**: {{DOI}}
- **ISSN**: {{ISSN}}
- **ISBN**: 978-0-13-301367-2
```
```ad-quote
title: Abstract
{{abstractNote}}
```
```ad-abstract
title: Files and Links
- **Url**: {{url}}
- **Uri**: http://zotero.org/users/7659667/items/DY8FYKM8
- **Eprint**: {{eprint}}
- **File**: [Jalote - 1994 - Fault tolerance in distributed systems.pdf](file://D:\Zotero\storage\XY2VCDKG\Jalote%20-%201994%20-%20Fault%20tolerance%20in%20distributed%20systems.pdf)
- **Local Library**: [Zotero]((zotero://select/library/items/DY8FYKM8))
```
```ad-note
title: Tags and Collections
- **Keywords**: Distributed Processing; Fault Tolerance
- **Collections**: Books; Fault Tolerance
```

----

## Comments



----

## Extracted Annotations

Annotations(5/8/2022, 11:26:33 AM)

- *“External clock synchronization requires maintaining processor clocks within some given maximum deviation from an external time reference, which keeps real time. Internal clock synchronization requires that the clocks of different processors be kept within some maximum relative deviation of each other.”* [(Jalote, 1994, p. 102)](zotero://open-pdf/library/items/XY2VCDKG?page=102&annotation=B4FJHCBR)

- *“Externally synchronized clocks are also internally synchronized, though the reverse is not true.”* [(Jalote, 1994, p. 102)](zotero://open-pdf/library/items/XY2VCDKG?page=102&annotation=UMBRDNBG)

- *“This implies that the network delay has to be assessed properly if the clock values of other nodes are to be determined.”* [(Jalote, 1994, p. 102)](zotero://open-pdf/library/items/XY2VCDKG?page=102&annotation=M6CZA4ZC)

- *“a clock may be “dual-faced”* [(Jalote, 1994, p. 103)](zotero://open-pdf/library/items/XY2VCDKG?page=103&annotation=KA2ZTFYQ)>  **shortcoming of clock**

- *“Let Ci (t ) denote the reading of a clock Q (i.e., the value returned by the process controlling this clock if an attempt is made to read this clock) at the physical time t . Let Ci (T ) denote the real time when the ith clock reaches a value T .”* [(Jalote, 1994, p. 103)](zotero://open-pdf/library/items/XY2VCDKG?page=103&annotation=T3G4JP22)

>  **non-faulty clock**

![](file://D:\Zotero\storage\XYIQDRMZ\image.png)[ ](zotero://open-pdf/library/items/XY2VCDKG?page=103&annotation=AFF5R49C)

- *“We assume that a process can read a clock only by sending a message to the process controlling that clock.”* [(Jalote, 1994, p. 103)](zotero://open-pdf/library/items/XY2VCDKG?page=103&annotation=MDKPEGBN)

- *“In deterministic protocols, the clock synchronization conditions and the bounds are guaranteed. However, these protocols often require some assumptions about message delays. Probabilistic clock synchronization does not require any assumption about maximum message delays, but guarantees precision only with a probability.”* [(Jalote, 1994, p. 104)](zotero://open-pdf/library/items/XY2VCDKG?page=104&annotation=A5KMAB9Y)>  **differences between deterministic and probabilistic protocols**

- *“One approach to synchronization isthat a processreads all the clocksin the system and then setsits value to the median of these values.”* [(Jalote, 1994, p. 104)](zotero://open-pdf/library/items/XY2VCDKG?page=104&annotation=AY6QHFQ3)

- *“However, due to the possibility of dual-faced clocks, the directly read value of a clock cannot be used.”* [(Jalote, 1994, p. 104)](zotero://open-pdf/library/items/XY2VCDKG?page=104&annotation=VPARNNKH)

- *“If max is the maximum time delay for message delivery, and min is the minimum time delay, and there are n processes, then for a deterministic clock synchronization protocol, the closeness of synchronization that can be attained is roughly (max — min )(1 — l / n) [LL88].”* [(Jalote, 1994, p. 105)](zotero://open-pdf/library/items/XY2VCDKG?page=105&annotation=ERGKXYGG)

- *“assuming that the number of faulty processesisless than n/3 for an n processsystem. It requires only n2 messagesin a synchronization round”* [(Jalote, 1994, p. 105)](zotero://open-pdf/library/items/XY2VCDKG?page=105&annotation=2JV3ZDER)

- *“the hardware clock is never altered, and that the drift of a correct clock is bounded by p”* [(Jalote, 1994, p. 105)](zotero://open-pdf/library/items/XY2VCDKG?page=105&annotation=YC429TV3)

- *“H (t ) is the time shown by the hardware clock at time t , and CORR(t ) is the adjustment function, whose value changes with time”* [(Jalote, 1994, p. 105)](zotero://open-pdf/library/items/XY2VCDKG?page=105&annotation=TIZM4ZNH)

- *“the number of faulty processes is at most /,such that the total number of processes in the system is at least 3/ + 1.”* [(Jalote, 1994, p. 106)](zotero://open-pdf/library/items/XY2VCDKG?page=106&annotation=P28P6AZY)

- *“A message is delivered in [5 — e, 5 + e ] time, for a fixed S and e , with &lt;$ &gt; &lt;?.”* [(Jalote, 1994, p. 106)](zotero://open-pdf/library/items/XY2VCDKG?page=106&annotation=MJN9QBEZ)

![](file://D:\Zotero\storage\64ZN8QNT\image.png)[ ](zotero://open-pdf/library/items/XY2VCDKG?page=107&annotation=JZU9KH5F)

- *“The main parameters of the algorithm are p (bound on the clock drift), f) (bound on how far apart the clocks are initially), 8 , (bound on message delays), and AT (period between rounds).”* [(Jalote, 1994, p. 107)](zotero://open-pdf/library/items/XY2VCDKG?page=107&annotation=HZ9UE6P3)

![](file://D:\Zotero\storage\ZKQSSV6A\image.png)[ ](zotero://open-pdf/library/items/XY2VCDKG?page=110&annotation=X5J24TXZ)

![](file://D:\Zotero\storage\NMWRVZ63\image.png)[ ](zotero://open-pdf/library/items/XY2VCDKG?page=110&annotation=YMM7ZFAN)

![](file://D:\Zotero\storage\9AGMIY3W\image.png)[ ](zotero://open-pdf/library/items/XY2VCDKG?page=110&annotation=FI8YN2D7)

- *“[Cri89]”* [(Jalote, 1994, p. 413)](zotero://open-pdf/library/items/XY2VCDKG?page=413&annotation=YDWWLDNB)

- *“[DHS84]”* [(Jalote, 1994, p. 413)](zotero://open-pdf/library/items/XY2VCDKG?page=413&annotation=BGQPB4WL)

- *“[LL88]”* [(Jalote, 1994, p. 419)](zotero://open-pdf/library/items/XY2VCDKG?page=419&annotation=U3JEYAG4)

- *“[LM85]”* [(Jalote, 1994, p. 420)](zotero://open-pdf/library/items/XY2VCDKG?page=420&annotation=33TJFR2P)

- *“[LSP82]”* [(Jalote, 1994, p. 420)](zotero://open-pdf/library/items/XY2VCDKG?page=420&annotation=Y8SYNCIR)

