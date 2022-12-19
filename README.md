# IATBDP
repository per codici per il corso di introduzione alla teoria bayesiana della probabilità

Scritti insieme a Carlo Panu

#Breve spiegazione dei codci

##fit\_pol\_ran.py

Codice che prova a fittare dei dati, contenuti in data.txt con un modello polinomiale cercando i valori ottimali della likelihood calcaloata in un set di parametri estratti uniformemente in un range scelto dall'utente.

##lupi.py

Anvendo visto M lupi, taggati r... non mi ricordo CARLO COMPLETALO TU

##metropolis_hastings.py

Campionamento di una gaussiana tramite algoritmo metropolis-hastings

##SIRD.py

simulazione modello sird

##nested_sampling.py

Implementazione del nested sampling per integrare una gaussiana D dimensionale. Leggere attentamente i commenti ci sono dei caveat

##nested_sampling.jl

Non vi sorprenderà sapere che è lo stesso di quello sopra ma scritto in julia. Questo perchè il tempo impiegato da julia è nettamente inferiore (Nel caso peggiore, 30 minuti vs 20 secondi) Me cojoni come dicono i saggi. Leggere attentamente i commenti ci sono dei caveat, tranquilli sono gli stessi di sopra.

##plot_hit.py

Permette di realizzare i grafici, contenuti nella cartella plot\_D_\50 leggendo i dati creati dal codice julia. Voi direte: ma julia può fare i plot da sè, e avete pure ragione, ma a me non piace
