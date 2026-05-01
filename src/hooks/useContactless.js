import { useEffect, useRef, useCallback, useState } from 'react';
import { FilesetResolver, GestureRecognizer } from '@mediapipe/tasks-vision';

export const useContactless = (callback, { runningMode = 'VIDEO', numHands = 2, swipeThreshold = 0.10 } = {}) => {
  const recognizerRef = useRef(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const isRunning = useRef(false);
  const rafIdRef = useRef(null);
  const callbackRef = useRef(callback);
  const [isReady, setIsReady] = useState(false);

  // Keep callbackRef current without it being a dep of any useCallback
  useEffect(() => {
    callbackRef.current = callback;
  });

  // Advanced gesture tracker state
  const tracking = useRef({
    historyX: [],
    historyY: [],
    swipeAccumulator: 0,
    lastGestureName: '',
    lastOpenPalmTime: 0,
    cooldownEnd: 0,
    smoothedX: null
  });

  const initHandTracking = useCallback(async () => {
    try {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
      );
      const recognizer = await GestureRecognizer.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
          delegate: "GPU"
        },
        runningMode: runningMode,
        numHands: numHands
      });
      recognizerRef.current = recognizer;
      setIsReady(true);
    } catch (e) {
      console.error("Failed to initialize MediaPipe", e);
    }
  }, [numHands, runningMode]);

  const stopCamera = useCallback(() => {
    isRunning.current = false;
    if (rafIdRef.current !== null) {
      cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  const processComplexGestures = useCallback((results) => {
    if (!results.landmarks || results.landmarks.length === 0) {
       tracking.current.historyX = [];
       tracking.current.historyY = [];
       tracking.current.swipeAccumulator = 0;
       tracking.current.lastGestureName = '';
       tracking.current.smoothedX = null;
       return null;
    }

    const now = Date.now();
    let semanticGesture = null;

    if (now > tracking.current.cooldownEnd) {
      
      // Single hand Gestures mapping
      const gestureObj = results.gestures && results.gestures.length > 0 ? results.gestures[0][0] : null;
      const currentName = gestureObj ? gestureObj.categoryName : 'None';

      if (currentName !== tracking.current.lastGestureName) {
        switch (currentName) {
          case 'Closed_Fist':
            semanticGesture = 'FIST';
            tracking.current.cooldownEnd = now + 500;
            break;
          case 'Pointing_Up':
            semanticGesture = 'ONE_FINGER';
            tracking.current.cooldownEnd = now + 500;
            break;
          case 'Victory':
            semanticGesture = 'TWO_FINGERS';
            tracking.current.cooldownEnd = now + 500;
            break;
          case 'Thumb_Up':
            semanticGesture = 'THUMBS_UP';
            tracking.current.cooldownEnd = now + 500;
            break;
          default:
            break;
        }
      }
      
      // 3. Simple & Robust Swipe Detection (4 directions) - ONLY with Open_Palm
      const rawX = results.landmarks[0][0].x; 
      const rawY = results.landmarks[0][0].y; 
      
      tracking.current.historyX.push({ val: rawX, time: now });
      tracking.current.historyY.push({ val: rawY, time: now });
      if (tracking.current.historyX.length > 15) tracking.current.historyX.shift();
      if (tracking.current.historyY.length > 15) tracking.current.historyY.shift();

      if (tracking.current.historyX.length >= 3 && !semanticGesture && currentName === 'Open_Palm') {
         const oldX = tracking.current.historyX[0];
         const oldY = tracking.current.historyY[0];
         const deltaX = rawX - oldX.val;
         const deltaY = rawY - oldY.val;
         const deltaTime = now - oldX.time;
         
         // If a quick movement occurs (under 400ms)
         if (deltaTime > 0 && deltaTime < 400) { 
             const absX = Math.abs(deltaX);
             const absY = Math.abs(deltaY);
             
             // Check if movement is significant enough
             if (absX > swipeThreshold || absY > swipeThreshold) {
                // Determine dominant direction
                if (absX > absY) {
                   // Horizontal swipe
                   semanticGesture = deltaX > 0 ? 'SWIPE_RIGHT' : 'SWIPE_LEFT'; 
                } else {
                   // Vertical swipe
                   semanticGesture = deltaY > 0 ? 'SWIPE_DOWN' : 'SWIPE_UP'; 
                }
                
                tracking.current.historyX = [];
                tracking.current.historyY = [];
                tracking.current.cooldownEnd = now + 700;
             }
         }
      }

      tracking.current.lastGestureName = currentName;
    }

    return semanticGesture;
  }, [swipeThreshold]);

  const predictWebcam = useCallback(() => {
    if (!recognizerRef.current || !videoRef.current) return;

    // Cancel any existing loop before starting a new one
    if (rafIdRef.current !== null) {
      cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
    }

    isRunning.current = true;
    let lastVideoTime = -1;

    const tick = () => {
      if (!isRunning.current) return;

      if (videoRef.current && videoRef.current.currentTime !== lastVideoTime && videoRef.current.videoWidth > 0 && videoRef.current.videoHeight > 0) {
        lastVideoTime = videoRef.current.currentTime;
        try {
          const results = recognizerRef.current.recognizeForVideo(videoRef.current, performance.now());
          const semanticGesture = processComplexGestures(results);

          // Use ref so callback changes never force a new loop
          if (callbackRef.current) {
            callbackRef.current({ ...results, semanticGesture });
          }
        } catch (err) {
          console.error("Inference Error:", err);
        }
      }
      rafIdRef.current = requestAnimationFrame(tick);
    };
    tick();
  }, [processComplexGestures]);

  const startCamera = useCallback(async (videoElement) => {
    videoRef.current = videoElement;
    if (!videoElement) return;

    // Guard: don't restart if a stream is already live
    if (streamRef.current) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (!videoRef.current) return; // component may have unmounted during await
      videoRef.current.srcObject = stream;
      streamRef.current = stream;
      videoRef.current.onloadedmetadata = () => {
        videoRef.current.play();
        predictWebcam();
      };
    } catch (e) {
      console.error("Error accessing camera", e);
    }
  }, [predictWebcam]);

  useEffect(() => {
    const init = async () => {
      await initHandTracking();
    };
    init();
    return () => stopCamera(); 
  }, [initHandTracking, stopCamera]);

  return { start: startCamera, stop: stopCamera, isReady };
};
