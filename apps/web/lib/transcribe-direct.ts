/**
 * Direct transcription without workflow queue.
 * Bypasses the Vercel workflow system to avoid ArrayBuffer serialization issues.
 */

import { promises as fs } from "node:fs";
import { db } from "@cap/database";
import { organizations, s3Buckets, users, videos } from "@cap/database/schema";
import { serverEnv } from "@cap/env";
import { userIsPro } from "@cap/utils";
import { S3Buckets } from "@cap/web-backend";
import type { S3Bucket, Video } from "@cap/web-domain";
import { createClient } from "@deepgram/sdk";
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
import { type DeepgramResult, formatToWebVTT } from "@/lib/transcribe-utils";

interface TranscribeDirectPayload {
	videoId: string;
	userId: string;
	aiGenerationEnabled: boolean;
}

export async function transcribeVideoDirect(
	payload: TranscribeDirectPayload,
): Promise<{ success: boolean; message: string }> {
	const { videoId, userId, aiGenerationEnabled } = payload;

	console.log(`[transcribe-direct] Starting transcription for video ${videoId}`);

	try {
		// Validate and get video data
		if (!serverEnv().DEEPGRAM_API_KEY) {
			throw new Error("Missing DEEPGRAM_API_KEY");
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

		// Upload temp audio
		const audioKey = `${userId}/${videoId}/audio-temp.mp3`;
		await bucket
			.putObject(audioKey, audioBuffer, { contentType: "audio/mpeg" })
			.pipe(runPromise);

		const audioSignedUrl = await bucket
			.getSignedObjectUrl(audioKey)
			.pipe(runPromise);

		// Transcribe with Deepgram
		console.log(`[transcribe-direct] Sending to Deepgram...`);
		const audioResponse = await fetch(audioSignedUrl);
		if (!audioResponse.ok) {
			throw new Error(`Audio URL not accessible: ${audioResponse.status}`);
		}

		const audioArrayBuffer = await audioResponse.arrayBuffer();
		const audioBufferForDG = Buffer.from(audioArrayBuffer);

		const deepgram = createClient(serverEnv().DEEPGRAM_API_KEY as string);

		const { result: dgResult, error } = await deepgram.listen.prerecorded.transcribeFile(
			audioBufferForDG,
			{
				model: "nova-3",
				smart_format: true,
				detect_language: true,
				utterances: true,
				mime_type: "audio/mpeg",
			},
		);

		if (error) {
			throw new Error(`Deepgram transcription failed: ${error.message}`);
		}

		const transcription = formatToWebVTT(dgResult as unknown as DeepgramResult);

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

		// Queue AI generation if enabled
		if (aiGenerationEnabled) {
			await startAiGeneration(videoId as Video.VideoId, userId);
		}

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
