"""
services/speaker_detector.py
─────────────────────────────
Speaker detection — now consumes the DENSE per-frame MAR data produced by
multi_face_tracker (30fps lip analysis) instead of running its own video pass.

IMPROVED: Better speaker-to-face mapping with temporal windowing and multi-modal fusion.
"""

from __future__ import annotations

import os
import importlib
import logging
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Any ,Dict ,List ,Optional ,Tuple

import numpy as np

logger =logging .getLogger (__name__ )

try :
    _pyannote_audio =importlib .import_module ("pyannote.audio")
except ImportError :
    _pyannote_audio =None

try :
    _torch =importlib .import_module ("torch")
except ImportError :
    _torch =None

@dataclass
class SpeakerSegment :
    speaker_id :str
    start :float
    end :float
    face_id :Optional [int ]=None
    confidence :float =1.0
    # Energy-based VAD can tell us that speech is present, but it cannot tell
    # which person produced it.  Keep that distinction explicit so a generic
    # SPEAKER_00 label does not get permanently attached to one visible face.
    identity_reliable :bool =True

@dataclass
class WordWithSpeaker :
    text :str
    start :float
    end :float
    speaker_id :str
    face_id :Optional [int ]=None

def _extract_audio_wav (video_path :str ,out_wav :Optional [str ]=None )->str :
    created_temp =out_wav is None
    completed =False
    if out_wav is None :
        fd ,out_wav =tempfile .mkstemp (suffix =".wav")
        os .close (fd )

    cmd =[
    "ffmpeg","-y","-loglevel","error",
    "-i",video_path ,
    "-vn",
    "-acodec","pcm_s16le",
    "-ar","16000",
    "-ac","1",
    "-af","highpass=f=80,lowpass=f=8000",
    out_wav ,
    ]
    try :
        result =subprocess .run (cmd ,capture_output =True ,text =True ,timeout =120 )
        if result .returncode !=0 :
            raise RuntimeError (f"FFmpeg failed: {result .stderr }")
        completed =True
        return out_wav
    except subprocess .TimeoutExpired :
        raise RuntimeError ("Audio extraction timed out")
    finally :
        if created_temp and not completed and out_wav and os .path .exists (out_wav ):
            try :
                os .remove (out_wav )
            except OSError :
                pass

class AudioSpeakerDiarizer :
    def __init__ (
    self ,
    hf_token :Optional [str ]=None ,
    min_speakers :Optional [int ]=None ,
    max_speakers :Optional [int ]=None ,
    ):
        self .hf_token =hf_token or os .environ .get ("HUGGINGFACE_TOKEN","")
        self .min_speakers =min_speakers
        self .max_speakers =max_speakers
        self ._pipeline :Any =None

        if _pyannote_audio is not None and self .hf_token :
            try :
                self ._pipeline =_pyannote_audio .Pipeline .from_pretrained (
                "pyannote/speaker-diarization-3.1",
                use_auth_token =self .hf_token ,
                )
                if _torch is not None and _torch .cuda .is_available ():
                    self ._pipeline .to (_torch .device ("cuda"))
                    logger .info ("pyannote on CUDA")
                else :
                    logger .info ("pyannote on CPU")
            except Exception as exc :
                logger .warning ("pyannote load failed: %s — energy VAD fallback.",exc )
                self ._pipeline =None

    def diarize (self ,audio_path :str ,*,is_video :bool =False )->List [SpeakerSegment ]:
        wav_path =None
        cleanup =False
        try :
            if is_video :
                wav_path =_extract_audio_wav (audio_path )
                cleanup =True
                target =wav_path
            else :
                target =audio_path

            if self ._pipeline is not None :
                return self ._run_pyannote (target )
            return self ._run_energy_vad (target )
        finally :
            if cleanup and wav_path and os .path .exists (wav_path ):
                try :
                    os .remove (wav_path )
                except OSError :
                    pass

    def annotate_words (
    self ,words :List [Dict [str ,Any ]],segments :List [SpeakerSegment ]
    )->List [WordWithSpeaker ]:
        result =[]
        for w in words :
            mid =(w ["start"]+w ["end"])/2.0
            spk =self ._find_speaker (mid ,segments )
            face_id =None
            for seg in segments :
                if seg .start <=mid <=seg .end and seg .speaker_id ==spk :
                    face_id =seg .face_id
                    break
            result .append (WordWithSpeaker (
            text =w ["text"],start =w ["start"],end =w ["end"],
            speaker_id =spk ,face_id =face_id ,
            ))
        return result

    def _run_pyannote (self ,audio_path :str )->List [SpeakerSegment ]:
        kwargs :Dict [str ,Any ]={}
        if self .min_speakers is not None :
            kwargs ["min_speakers"]=self .min_speakers
        if self .max_speakers is not None :
            kwargs ["max_speakers"]=self .max_speakers
        try :
            dia =self ._pipeline (audio_path ,**kwargs )
            segs =[
            SpeakerSegment (speaker_id =spk ,start =turn .start ,end =turn .end )
            for turn ,_ ,spk in dia .itertracks (yield_label =True )
            ]
            return self ._merge_short (segs )
        except Exception as e :
            logger .error ("pyannote failed: %s",e )
            return []

    @staticmethod
    def _merge_short (segs :List [SpeakerSegment ],max_gap :float =0.5 )->List [SpeakerSegment ]:
        if not segs :
            return segs
        segs =sorted (segs ,key =lambda s :s .start )
        merged =[segs [0 ]]
        for s in segs [1 :]:
            last =merged [-1 ]
            if s .speaker_id ==last .speaker_id and (s .start -last .end )<max_gap :
                last .end =max (last .end ,s .end )
            else :
                merged .append (s )
        return merged

    def _run_energy_vad (self ,audio_path :str )->List [SpeakerSegment ]:
        try :
            import wave
            with wave .open (audio_path ,"rb")as wf :
                framerate =wf .getframerate ()
                raw =wf .readframes (wf .getnframes ())
            samples =np .frombuffer (raw ,dtype =np .int16 ).astype (np .float32 )/32768.0
        except Exception as exc :
            logger .warning ("VAD read failed: %s",exc )
            return []

        frame_len =int (framerate *0.025 )
        hop_len =int (framerate *0.010 )

        energies =[]
        t ,i =0.0 ,0
        while i +frame_len <=len (samples ):
            frame =samples [i :i +frame_len ]
            rms =float (np .sqrt (np .mean (frame **2 )))
            energies .append ((t ,rms ))
            i +=hop_len
            t +=0.010

        if not energies :
            return []

        rms_vals =[e [1 ]for e in energies ]
        noise_floor =np .percentile (rms_vals ,20 )
        threshold =max (0.02 ,noise_floor *3.0 )

        segs :List [SpeakerSegment ]=[]
        in_seg ,seg_start =False ,0.0
        for t ,rms in energies :
            if rms >threshold and not in_seg :
                in_seg ,seg_start =True ,t
            elif rms <=threshold and in_seg :
                in_seg =False
                if t -seg_start >0.2 :
                    segs .append (SpeakerSegment (
                    "SPEAKER_00",seg_start ,t ,identity_reliable =False
                    ))
        if in_seg and energies :
            segs .append (SpeakerSegment (
            "SPEAKER_00",seg_start ,energies [-1 ][0 ],identity_reliable =False
            ))

        return self ._merge_short (segs ,max_gap =0.3 )

    @staticmethod
    def _find_speaker (t :float ,segments :List [SpeakerSegment ])->str :
        for seg in segments :
            if seg .start <=t <=seg .end :
                return seg .speaker_id
        return "UNKNOWN"

def _rolling_variance (arr :np .ndarray ,window :int )->np .ndarray :
    out =np .zeros_like (arr )
    half =window //2
    for i in range (len (arr )):
        lo =max (0 ,i -half )
        hi =min (len (arr ),i +half +1 )
        out [i ]=float (np .var (arr [lo :hi ]))
    return out

def _robust_normalize (arr :np .ndarray )->np .ndarray :
    p10 ,p90 =np .percentile (arr ,[10 ,90 ])
    rng =max (1e-6 ,p90 -p10 )
    return np .clip ((arr -p10 )/rng ,0 ,1 )

def _sustain_filter (binary :np .ndarray ,min_run :int )->np .ndarray :
    result =binary .copy ()

    i =0
    while i <len (result ):
        if result [i ]:
            j =i
            while j <len (result )and result [j ]:
                j +=1
            if j -i <min_run :
                result [i :j ]=False
            i =j
        else :
            i +=1

    i =0
    while i <len (result ):
        if not result [i ]:
            j =i
            while j <len (result )and not result [j ]:
                j +=1
            if 0 <i and j <len (result )and (j -i )<max (1 ,min_run //2 ):
                result [i :j ]=True
            i =j
        else :
            i +=1
    return result

def mar_series_to_segments (
times :np .ndarray ,
mars :np .ndarray ,
sample_interval :float ,
mar_threshold :float =0.3 ,
min_seg_duration :float =0.25 ,
)->List [Tuple [float ,float ]]:
    """Convert a (time, MAR) series into speaking intervals."""
    if len (times )<5 :
        return []

    window =max (3 ,int (0.4 /max (0.01 ,sample_interval )))
    mar_var =_rolling_variance (mars ,window )
    score =_robust_normalize (mar_var )

    noise =np .percentile (score ,25 )
    threshold =max (mar_threshold ,noise +0.15 )

    raw =score >threshold
    min_frames =max (2 ,int (min_seg_duration /max (0.01 ,sample_interval )))
    speaking =_sustain_filter (raw ,min_frames )

    segs :List [Tuple [float ,float ]]=[]
    in_seg ,start =False ,0.0
    for i ,s in enumerate (speaking ):
        if s and not in_seg :
            in_seg ,start =True ,times [i ]
        elif not s and in_seg :
            in_seg =False
            if times [i ]-start >=min_seg_duration :
                segs .append ((float (start ),float (times [i ])))
    if in_seg :
        if times [-1 ]-start >=min_seg_duration :
            segs .append ((float (start ),float (times [-1 ])))

    merged :List [Tuple [float ,float ]]=[]
    for s in segs :
        if merged and s [0 ]-merged [-1 ][1 ]<0.3 :
            merged [-1 ]=(merged [-1 ][0 ],s [1 ])
        else :
            merged .append (s )
    return merged

def _visual_from_dense_timeline (face_timeline :List )->Dict [int ,List [Tuple [float ,float ]]]:
    """
    Build per-face_id speaking intervals from DENSE timeline MAR data.
    Returns {face_id: [(start, end), ...]} in clip-relative time.
    """
    series :Dict [int ,Tuple [List [float ],List [float ]]]={}
    for fd in face_timeline :
        for face in fd .faces :
            if face .is_coasted :
                continue
            if face .face_id not in series :
                series [face .face_id ]=([],[])
            t ,m =series [face .face_id ]
            t .append (fd .timestamp )
            m .append (face .mar )
            series [face .face_id ]=(t ,m )

    out :Dict [int ,List [Tuple [float ,float ]]]={}
    for fid ,(ts ,ms )in series .items ():
        if len (ts )<10 :
            continue
        t_arr =np .array (ts )
        m_arr =np .array (ms )

        if len (t_arr )>1 :
            dt =float (np .median (np .diff (t_arr )))
        else :
            dt =1 /30.0
        segs =mar_series_to_segments (t_arr ,m_arr ,sample_interval =max (dt ,1 /60.0 ))
        if segs :
            out [fid ]=segs
    return out

class SpeakerDetector :
    """
    Priority order for face assignment:
      1. Dense MAR from face_timeline (best — 30fps, same pass as tracking)
      2. Audio-only area voting (when no face timeline)

    IMPROVED: Better temporal windowing and multi-modal fusion for accurate face assignment.
    """

    def __init__ (
    self ,
    hf_token :Optional [str ]=None ,
    use_visual :bool =True ,
    sample_interval :float =0.1 ,
    max_speakers :Optional [int ]=None ,
    ):
        self .audio_diarizer =AudioSpeakerDiarizer (
        hf_token =hf_token ,max_speakers =max_speakers ,
        )
        self .use_visual =use_visual

    def detect (
    self ,
    video_path :str ,
    face_timeline :Optional [List ]=None ,
    all_words :Optional [List [Dict ]]=None ,
    clip_start :float =0.0 ,
    clip_end :Optional [float ]=None ,
    )->List [SpeakerSegment ]:

        audio_segs =self .audio_diarizer .diarize (video_path ,is_video =True )

        if clip_end is not None :
            audio_segs =[
            s for s in audio_segs
            if s .end >clip_start and s .start <clip_end
            ]
        for seg in audio_segs :
            seg .start =max (0.0 ,seg .start -clip_start )
            seg .end =seg .end -clip_start
            if clip_end :
                seg .end =min (seg .end ,clip_end -clip_start )

        if face_timeline :
            has_dense_mar =any (
            getattr (f ,"mar",0.0 )>0.0
            for fd in face_timeline [:50 ]
            for f in fd .faces
            )
            if self .use_visual and has_dense_mar :
                visual =_visual_from_dense_timeline (face_timeline )
                self ._assign_with_visual (audio_segs ,visual ,face_timeline )
                logger .info (
                "[Speaker Detection] dense-MAR visual intervals for %d faces",
                len (visual ),
                )
            else :
                self ._assign_by_area (audio_segs ,face_timeline )

        return sorted (audio_segs ,key =lambda s :s .start )

    def annotate_words (
    self ,words :List [Dict ],segments :List [SpeakerSegment ]
    )->List [WordWithSpeaker ]:
        return self .audio_diarizer .annotate_words (words ,segments )

    def close (self )->None :
        pass

    def _assign_with_visual (
    self ,
    audio_segs :List [SpeakerSegment ],
    visual :Dict [int ,List [Tuple [float ,float ]]],
    face_timeline :List ,
    )->None :
        """
        Enhanced face assignment with temporal windowing and multi-modal fusion.
        Score faces per audio segment: visual speaking overlap >> area >> age.
        """
        for seg in audio_segs :
            # VAD segments contain speech timing only.  Assigning one of them
            # to a face would make the active reframer strongly prefer that
            # face for the whole utterance, even when another person speaks.
            # The reframer uses dense, short-window mouth activity instead.
            if not seg .identity_reliable :
                seg .face_id =None
                continue

            seg_dur =max (1e-6 ,seg .end -seg .start )

            padding =0.2
            frames_in_seg =[
            fd for fd in face_timeline
            if (seg .start -padding )<=fd .timestamp <=(seg .end +padding )
            ]

            if not frames_in_seg :
                continue

            scores :Dict [int ,Dict [str ,float ]]=defaultdict (lambda :{
            'visual_speaking':0.0 ,
            'area':0.0 ,
            'presence':0.0 ,
            'mar_peak':0.0 ,
            })

            for fd in frames_in_seg :

                t_normalized =(fd .timestamp -seg .start )/seg_dur if seg_dur >0 else 0.5
                temporal_weight =1.0 -abs (0.5 -t_normalized )*0.5

                for face in fd .faces :
                    if face .is_coasted :
                        continue

                    fid =face .face_id

                    intervals =visual .get (fid ,[])
                    for vs ,ve in intervals :
                        if vs <=fd .timestamp <=ve :

                            scores [fid ]['visual_speaking']+=15.0 *temporal_weight
                            break

                    if face .mar >0.25 :
                        scores [fid ]['mar_peak']+=face .mar *8.0 *temporal_weight

                    area_norm =min (face .box .area /100000.0 ,1.0 )
                    scores [fid ]['area']+=area_norm *face .box .confidence *3.0 *temporal_weight

                    scores [fid ]['presence']+=1.0 *temporal_weight

            final_scores ={}
            for fid ,components in scores .items ():
                total =sum (components .values ())

                if components ['visual_speaking']>10.0 :
                    total *=1.5

                presence_ratio =components ['presence']/len (frames_in_seg )if frames_in_seg else 0
                if presence_ratio <0.3 :
                    total *=0.5

                final_scores [fid ]=total

            if final_scores :
                ranked =sorted (final_scores .items (),key =lambda kv :kv [1 ],reverse =True )
                seg .face_id =ranked [0 ][0 ]

                if len (ranked )>1 :
                    score_ratio =ranked [0 ][1 ]/(ranked [1 ][1 ]+1e-6 )
                    seg .confidence =float (np .clip (score_ratio /3.0 ,0.4 ,1.0 ))
                else :
                    seg .confidence =1.0

                if scores [seg .face_id ]['visual_speaking']>20.0 :
                    seg .confidence =min (1.0 ,seg .confidence *1.3 )

        self ._enforce_speaker_consistency (
        [seg for seg in audio_segs if seg .identity_reliable ]
        )

    def _enforce_speaker_consistency (self ,audio_segs :List [SpeakerSegment ])->None :
        """
        Ensure one speaker_id consistently maps to one face_id across the timeline.
        Uses weighted voting with BIJECTIVE (1-to-1) constraint.

        FIXED: Prevents multiple speakers from mapping to same face_id.
        """

        votes :Dict [str ,Dict [int ,float ]]=defaultdict (lambda :defaultdict (float ))

        for i ,seg in enumerate (audio_segs ):
            if seg .face_id is not None :
                duration =seg .end -seg .start
                weight =duration *seg .confidence

                recency_factor =1.0 +(i /max (len (audio_segs ),1 ))*0.2

                votes [seg .speaker_id ][seg .face_id ]+=weight *recency_factor

        candidates =[]
        for spk ,face_votes in votes .items ():
            for fid ,score in face_votes .items ():
                candidates .append ((spk ,fid ,score ))

        candidates .sort (key =lambda x :x [2 ],reverse =True )

        used_faces =set ()
        canonical_mapping :Dict [str ,int ]={}

        for spk ,fid ,score in candidates :
            if spk not in canonical_mapping and fid not in used_faces :
                canonical_mapping [spk ]=fid
                used_faces .add (fid )
                logger .debug (f"Canonical mapping: {spk } → Face {fid } (score={score :.2f})")

        for seg in audio_segs :
            if seg .speaker_id in canonical_mapping :
                canonical_fid =canonical_mapping [seg .speaker_id ]

                if (seg .face_id is None or
                seg .confidence <0.6 or
                (seg .face_id !=canonical_fid and seg .confidence <0.8 )):

                    seg .face_id =canonical_fid

                    seg .confidence =max (seg .confidence ,0.7 )

    def _assign_by_area (self ,audio_segs :List [SpeakerSegment ],face_timeline :List )->None :
        """Fallback: largest face wins per segment (single-speaker heuristic)."""
        votes :Dict [str ,Dict [int ,float ]]=defaultdict (lambda :defaultdict (float ))
        for fd in face_timeline :
            for seg in audio_segs :
                if not seg .identity_reliable :
                    continue
                if seg .start <=fd .timestamp <=seg .end :
                    for face in fd .faces :
                        votes [seg .speaker_id ][face .face_id ]+=face .box .area
        mapping ={
        spk :max (fv ,key =lambda fid :fv [fid ])
        for spk ,fv in votes .items ()if fv
        }
        for seg in audio_segs :
            if seg .identity_reliable :
                seg .face_id =mapping .get (seg .speaker_id )
            else :
                seg .face_id =None

def detect_speakers (
video_path :str ,
face_timeline :Optional [List ]=None ,
all_words :Optional [List [Dict ]]=None ,
clip_start :float =0.0 ,
clip_end :Optional [float ]=None ,
hf_token :Optional [str ]=None ,
use_visual :bool =True ,
)->Tuple [List [SpeakerSegment ],Optional [List [WordWithSpeaker ]]]:
    detector =SpeakerDetector (hf_token =hf_token ,use_visual =use_visual )
    try :
        segments =detector .detect (
        video_path =video_path ,
        face_timeline =face_timeline ,
        all_words =all_words ,
        clip_start =clip_start ,
        clip_end =clip_end ,
        )
        annotated =detector .annotate_words (all_words ,segments )if all_words else None
        return segments ,annotated
    finally :
        detector .close ()