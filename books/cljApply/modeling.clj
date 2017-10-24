(def earth {:name "Earth"
            :moons 1
            :volume 1.08321e12          ; km^3
            :mass 5.97219e24            ; kg
            :aphelion 152098232         ; km, farthest from sun
            :perihelion 147098290       ; km, closest from sun
            :type :Planet               ; entity type
            })

;;; another representation, type is obvious, record is efficient
(defrecord Planet [name
                   moons
                   volume
                   mass
                   aphelion
                   perihelion
                   ])

;;; two factory function is ready
;;; Positional factory 
(def earth 
  (->Planet "Earth" 1 1.08321e12 5.97219e24 152098232 147098290))

;;; Map factory 
(def earth 
  (map->Planet {:name "Earth"
                :moons 1
                :volume 1.08321e12
                :mass 5.97219e24
                :aphelion 152098232
                :perihelion 147098290}))



