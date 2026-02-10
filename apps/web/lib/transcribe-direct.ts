/**
 * Direct transcription using AssemblyAI with speaker diarization.
 * Bypasses the Vercel workflow system to avoid ArrayBuffer serialization issues.
 */

import { promises as fs } from "node:fs";
import { db } from "@cap/database";
import { organizations, s3Buckets, users, videos } from "@cap/database/schema";
import { serverEnv } from "@cap/env";
import { userIsPro } from "@cap/utils";
import { S3Buckets } from "@cap/web-backend";
import type { S3Bucket, Video } from "@cap/web-domain";
import { eq } from "drizzle-orm";
import { Option } from "effect";
import { checkHasAudioTrack, extractAudioFromUrl } from "@/lib/audio-extract";
import { startAiGeneration } from "@/lib/generate-ai";
import {
	checkHasAudioTrackViaMediaServer,
	extractAudioViaMediaServer,
	isMediaServerConfigured,
} from "@/lib/media-client";
import { runPromise } from "@/lib/server";

interface TranscribeDirectPayload {
	videoId: string;
	userId: string;
	aiGenerationEnabled: boolean;
}

interface AssemblyAIUtterance {
	speaker: string;
	text: string;
	start: number;
	end: number;
}

interface AssemblyAIResult {
	id: string;
	status: string;
	text: string;
	utterances?: AssemblyAIUtterance[];
	error?: string;
}

// Convert AssemblyAI result to WebVTT format with speaker labels
function formatToWebVTT(result: AssemblyAIResult): string {
	const lines: string[] = ["WEBVTT", ""];
	
	if (!result.utterances || result.utterances.length === 0) {
		// Fallback to plain text if no utterances
		lines.push("00:00:00.000 --> 00:00:10.000");
		lines.push(result.text || "");
		return lines.join("\n");
	}

	for (const utterance of result.utterances) {
		const startTime = formatTime(utterance.start);
		const endTime = formatTime(utterance.end);
		lines.push(`${startTime} --> ${endTime}`);
		lines.push(`<v Speaker ${utterance.speaker}>${utterance.text}`);
		lines.push("");
	}

	return lines.join("\n");
}

function formatTime(ms: number): string {
	const totalSeconds = Math.floor(ms / 1000);
	const hours = Math.floor(totalSeconds / 3600);
	const minutes = Math.floor((totalSeconds % 3600) / 60);
	const seconds = totalSeconds % 60;
	const milliseconds = ms % 1000;
	return `${hours.toString().padStart(2, "0")}:${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}.${milliseconds.toString().padStart(3, "0")}`;
}

async function transcribeWithAssemblyAI(audioUrl: string): Promise<string> {
	const apiKey = serverEnv().ASSEMBLYAI_API_KEY;
	if (!apiKey) {
		throw new Error("Missing ASSEMBLYAI_API_KEY");
	}

	const headers = {
		"authorization": apiKey,
		"content-type": "application/json",
	};

	// Submit transcription job with speaker diarization
	console.log(`[transcribe-direct] Submitting to AssemblyAI...`);
	const submitResponse = await fetch("https://api.assemblyai.com/v2/transcript", {
		method: "POST",
		headers,
		body: JSON.stringify({
			audio_url: audioUrl,
			speaker_labels: true,  // Enable speaker diarization
		}),
	});

	if (!submitResponse.ok) {
		const error = await submitResponse.text();
		throw new Error(`AssemblyAI submit failed: ${error}`);
	}

	const job = await submitResponse.json() as { id: string };
	console.log(`[transcribe-direct] AssemblyAI job ID: ${job.id}`);

	// Poll for completion
	let result: AssemblyAIResult;
	while (true) {
		await new Promise(resolve => setTimeout(resolve, 3000));
		
		const pollResponse = await fetch(`https://api.assemblyai.com/v2/transcript/${job.id}`, {
			headers: { authorization: apiKey },
		});

		if (!pollResponse.ok) {
			throw new Error(`AssemblyAI poll failed: ${pollResponse.status}`);
		}

		result = await pollResponse.json() as AssemblyAIResult;
		console.log(`[transcribe-direct] AssemblyAI status: ${result.status}`);

		if (result.status === "completed") {
			break;
		} else if (result.status === "error") {
			throw new Error(`AssemblyAI error: ${result.error}`);
		}
	}

	return formatToWebVTT(result);
}

export async function transcribeVideoDirect(
	payload: TranscribeDirectPayload,
): Promise<{ success: boolean; message: string }> {
	const { videoId, userId, aiGenerationEnabled } = payload;

	console.log(`[transcribe-direct] Starting transcription for video ${videoId}`);

	try {
		// Check for AssemblyAI key (primary) or Deepgram key (fallback)
		const assemblyKey = serverEnv().ASSEMBLYAI_API_KEY;
		if (!assemblyKey) {
			throw new Error("Missing ASSEMBLYAI_API_KEY");
		}

		const query = await db()
			.select({
				video: videos,
				bucket: s3Buckets,
				settings: videos.settings,
				orgSettings: organizations.settings,
				owner: users,
			})
			.from(videos)
			.leftJoin(s3Buckets, eq(videos.bucket, s3Buckets.id))
			.leftJoin(organizations, eq(videos.orgId, organizations.id))
			.innerJoin(users, eq(videos.ownerId, users.id))
			.where(eq(videos.id, videoId as Video.VideoId));

		if (query.length === 0) {
			throw new Error("Video does not exist");
		}

		const result = query[0];
		if (!result?.video) {
			throw new Error("Video information is missing");
		}

		const transcriptionDisabled =
			result.video.settings?.disableTranscript ??
			result.orgSettings?.disableTranscript ??
			false;

		if (transcriptionDisabled) {
			await db()
				.update(videos)
				.set({ transcriptionStatus: "SKIPPED" })
				.where(eq(videos.id, videoId as Video.VideoId));
			return { success: true, message: "Transcription disabled - skipped" };
		}

		// Mark as processing
		await db()
			.update(videos)
			.set({ transcriptionStatus: "PROCESSING" })
			.where(eq(videos.id, videoId as Video.VideoId));

		const bucketId = (result.bucket?.id ?? null) as S3Bucket.S3BucketId | null;

		// Extract audio
		const [bucket] = await S3Buckets.getBucketAccess(
			Option.fromNullable(bucketId),
		).pipe(runPromise);

		const videoKey = `${userId}/${videoId}/result.mp4`;
		const videoUrl = await bucket.getSignedObjectUrl(videoKey).pipe(runPromise);

		const response = await fetch(videoUrl, {
			method: "GET",
			headers: { range: "bytes=0-0" },
		});
		if (!response.ok) {
			throw new Error("Video file not accessible");
		}

		const useMediaServer = isMediaServerConfigured();
		let hasAudio: boolean;
		let audioBuffer: Buffer;

		if (useMediaServer) {
			console.log(`[transcribe-direct] Using media server for audio extraction`);
			hasAudio = await checkHasAudioTrackViaMediaServer(videoUrl);
			if (!hasAudio) {
				await db()
					.update(videos)
					.set({ transcriptionStatus: "NO_AUDIO" })
					.where(eq(videos.id, videoId as Video.VideoId));
				return { success: true, message: "Video has no audio track - skipped" };
			}
			audioBuffer = await extractAudioViaMediaServer(videoUrl);
		} else {
			console.log(`[transcribe-direct] Using local ffmpeg for audio extraction`);
			hasAudio = await checkHasAudioTrack(videoUrl);
			if (!hasAudio) {
				await db()
					.update(videos)
					.set({ transcriptionStatus: "NO_AUDIO" })
					.where(eq(videos.id, videoId as Video.VideoId));
				return { success: true, message: "Video has no audio track - skipped" };
			}

			const extractResult = await extractAudioFromUrl(videoUrl);
			try {
				audioBuffer = await fs.readFile(extractResult.filePath);
			} finally {
				await extractResult.cleanup();
			}
		}

		// Upload temp audio to S3 so AssemblyAI can access it
		const audioKey = `${userId}/${videoId}/audio-temp.mp3`;
		await bucket
			.putObject(audioKey, audioBuffer, { contentType: "audio/mpeg" })
			.pipe(runPromise);

		const audioSignedUrl = await bucket
			.getSignedObjectUrl(audioKey)
			.pipe(runPromise);

		// Transcribe with AssemblyAI (includes speaker diarization)
		const transcription = await transcribeWithAssemblyAI(audioSignedUrl);

		// Save transcription
		await bucket
			.putObject(`${userId}/${videoId}/transcription.vtt`, transcription, {
				contentType: "text/vtt",
			})
			.pipe(runPromise);

		await db()
			.update(videos)
			.set({ transcriptionStatus: "COMPLETE" })
			.where(eq(videos.id, videoId as Video.VideoId));

		// Cleanup temp audio
		try {
			await bucket.deleteObject(audioKey).pipe(runPromise);
		} catch (cleanupError) {
			console.error(`[transcribe-direct] Failed to cleanup temp audio:`, cleanupError);
		}

		// Skip AI generation for now (still has workflow queue issues)
		// if (aiGenerationEnabled) {
		// 	await startAiGeneration(videoId as Video.VideoId, userId);
		// }

		console.log(`[transcribe-direct] Transcription completed for video ${videoId}`);
		return { success: true, message: "Transcription completed successfully" };

	} catch (error) {
		console.error(`[transcribe-direct] Failed:`, error);
		
		// Reset status on failure
		await db()
			.update(videos)
			.set({ transcriptionStatus: null })
			.where(eq(videos.id, videoId as Video.VideoId));

		return {
			success: false,
			message: error instanceof Error ? error.message : "Transcription failed",
		};
	}
}
